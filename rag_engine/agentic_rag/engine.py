import re
import json
from rag_engine.agentic_rag.tools import LocalKnowledgeTool
from rag_engine.agentic_rag.prompts import AGENT_SYSTEM_PROMPT
from rag_engine.config_manager import ConfigManager
from rag_engine.history_manager import HistoryManager
# Import providers from simple_rag.engine or reimplement wrappers if needed
# To keep it DRY, we might want to abstract the LLM provider later, 
# but for now we will instantiate the providers similarly to SimpleRAGEngine.
import ollama
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

from rag_engine.model_manager import get_llm_path, is_gpu_available
# Llama-cpp import
from llama_cpp import Llama

class AgenticRAGEngine:
    def __init__(self, vector_store, embedder, provider: str, model_name: str, chat_id: str = None):
        self.vs = vector_store
        self.embedder = embedder
        self.provider = provider
        self.model_name = model_name
        self.chat_id = chat_id
        
        self.config = ConfigManager()
        self.history_manager = HistoryManager()
        
        # Initialize Tools
        self.knowledge_tool = LocalKnowledgeTool(self.vs, self.embedder)
        self.tools = [self.knowledge_tool]
        
        # Initialize LLM (Copying logic from SimpleRAGEngine)
        self.llm_local = None
        self.openai_client = None
        self.google_model = None
        
        self._init_llm()

    def _init_llm(self):
        if self.provider == "local":
            path = get_llm_path(self.model_name)
            n_gpu_layers = -1 if is_gpu_available() and self.config.get("use_gpu", True) else 0
            print(f"Loading local LLM (Agentic) from: {path}")
            self.llm_local = Llama(
                model_path=path,
                n_ctx=4096,
                n_threads=4,
                n_gpu_layers=n_gpu_layers,
                chat_format="llama-3",
                verbose=False
            )
        elif self.provider == "openai":
            if HAS_OPENAI:
                api_key = self.config.get("openai_api_key")
                self.openai_client = openai.OpenAI(api_key=api_key)
        elif self.provider == "google":
            if HAS_GOOGLE:
                api_key = self.config.get("google_api_key")
                genai.configure(api_key=api_key)
                self.google_model = genai.GenerativeModel(self.model_name or "gemini-pro")

    def _call_llm(self, messages, stop=None):
        """Helper to call the configured LLM synchronously."""
        if self.provider == "ollama":
            resp = ollama.chat(model=self.model_name, messages=messages, stream=False)
            return resp["message"]["content"]
        
        elif self.provider == "local":
            resp = self.llm_local.create_chat_completion(
                messages=messages,
                stop=stop,
                max_tokens=512,
                temperature=0.0 # Lower temp for reasoning
            )
            return resp["choices"][0]["message"]["content"]
            
        elif self.provider == "openai":
            if not self.openai_client:
                return "Error: OpenAI client not initialized."
            resp = self.openai_client.chat.completions.create(
                model=self.model_name or "gpt-3.5-turbo",
                messages=messages,
                stop=stop,
                temperature=0.0
            )
            return resp.choices[0].message.content
            
        elif self.provider == "google":
            if not HAS_GOOGLE:
                return "Error: Google client not initialized."
            # Gemini doesn't support system prompts in same way, nor stop sequences easily in some versions.
            # We will just append system prompt to user message if needed.
            # For brevity, implementing basic call.
            chat = self.google_model.start_chat(history=[])
            # Construct one big prompt from messages
            full_prompt = ""
            for m in messages:
                full_prompt += f"{m['role']}: {m['content']}\n"
            
            response = chat.send_message(full_prompt)
            return response.text
            
        return "Error: Provider not supported."

    def retrieve(self, query, similarity_method="cosine"):
        """
        In Agentic RAG, retrieval is handled BY the agent via tools.
        However, to maintain compatibility with the UI's `_stream_reply` which calls `retrieve` 
        before `chat`, we can return specific instructions or nothing.
        
        The UI expects `(context_str, context_results_list)`.
        
        We will return empty context here, because the Agent will decide if it needs to retrieve.
        Alternatively, we could do a pre-retrieval pass, but that defeats the purpose of being "Agentic".
        
        Let's return empty so the UI doesn't show a context bubble proactively.
        """
        return "", []

    def load_history(self):
        if self.chat_id:
            chat = self.history_manager.load_chat(self.chat_id)
            if chat:
                return chat.get("messages", [])
        return []

    def save_history(self, history):
        if self.chat_id:
            existing = self.history_manager.load_chat(self.chat_id)
            if existing:
                existing["messages"] = history
                self.history_manager.save_chat(existing)

    def chat(self, user_query, retrieved_context, stream=True):
        """
        Main ReAct Loop.
        Ignores `retrieved_context` because it fetches its own.
        Yields JSON events:
        {"message": {"content": "..."}} -> For final answer
        {"type": "thought", "content": "..."} -> For reasoning steps
        """
        
        # 1. Prepare Prompt
        tool_descriptions = "\n".join([f"{t.name}: {t.description}" for t in self.tools])
        tool_names = ", ".join([t.name for t in self.tools])
        
        system_prompt = AGENT_SYSTEM_PROMPT.format(
            tool_descriptions=tool_descriptions,
            tool_names=tool_names,
            input=user_query
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        max_steps = 5
        
        for step in range(max_steps):
            # Call LLM
            # We don't stream the reasoning generation from the LLM to avoid complex parsing of partial JSON/Thoughts.
            # We wait for the full "Thought... Action..." block, then yield it.
            
            llm_response = self._call_llm(messages, stop=["Observation:"])
            llm_response = llm_response.strip()
            
            # Yield the thought to UI
            # We try to extract just the Thought part if possible, or send the whole block
            yield {"type": "thought", "content": llm_response}
            
            # Append to history
            messages.append({"role": "assistant", "content": llm_response})
            
            # Check for Final Answer
            if "Final Answer:" in llm_response:
                final_answer = llm_response.split("Final Answer:")[-1].strip()
                yield {"message": {"content": final_answer}}
                return

            # Check for Action
            # Regex to find Action: tool_name AND Action Input: input
            action_match = re.search(r"Action:\s*(\w+)", llm_response)
            input_match = re.search(r"Action Input:\s*(.+)", llm_response)
            
            if action_match and input_match:
                tool_name = action_match.group(1).strip()
                tool_input = input_match.group(1).strip()
                
                # Execute Tool
                observation = f"Error: Tool {tool_name} not found."
                for tool in self.tools:
                    if tool.name == tool_name:
                        yield {"type": "thought", "content": f"🛠️ Executing {tool_name} with: {tool_input}..."}
                        observation = tool.run(tool_input)
                        yield {"type": "thought", "content": f"📝 Observation: {observation[:100]}..."}
                        break
                
                # Feed observation back
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                # If LLM didn't follow format or decided to stop without Final Answer
                if "Thought:" not in llm_response and "Action:" not in llm_response:
                     # Treat as final answer if it looks like conversation
                    yield {"message": {"content": llm_response}}
                    return
                
                # If it outputted a thought but no action, force it to continue
                # (Loop continues)

        yield {"message": {"content": "I reached my reasoning limit and could not find a final answer."}}
