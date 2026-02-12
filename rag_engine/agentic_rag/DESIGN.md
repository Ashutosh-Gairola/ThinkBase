# Agentic RAG Design & Implementation Process

## 1. Overview
Agentic RAG transforms a standard "Retrieve-then-Generate" pipeline into an autonomous system. Instead of blindly retrieving documents for every query, an **Agent** analyzes the request and decides:
1.  **Whether to retrieve** information at all (e.g., for greetings or general knowledge).
2.  **What to retrieve** (generating better search queries).
3.  **How to verify** the answer (checking if retrieved docs are relevant).
4.  **When to give up** or ask for clarification.

## 2. Architecture
The system will follow a **ReAct (Reasoning + Acting)** pattern or a **Router** pattern. Given the requirement for "Agentic RAG", a lightweight ReAct loop is recommended for maximum flexibility.

### Core Components
1.  **The Agent (Brain)**: The LLM (Ollama/OpenAI/Gemini) initialized with a system prompt that defines its persona and available tools.
2.  **The Tools (Capabilities)**:
    *   `VectorStoreTool`: Wraps your existing `vector_store.search()` to allow the agent to query the local knowledge base.
    *   `WebSearchTool` (Optional): To fetch live data if the local base is insufficient.
    *   `ContextTool` (Optional): To read specific files or history if needed.
3.  **The Executor (Loop)**: A loop that parses the LLM's output, detects "Actions", executes them, and feeds the "Observation" back to the LLM.

## 3. Implementation Process

### Phase 1: Define the Tool Interface
Create a standard interface for tools so the Agent can use them uniformly.

**File:** `rag_engine/agentic_rag/tools.py`
```python
class BaseTool:
    def __init__(self, name, description):
        self.name = name
        self.description = description

    def run(self, query: str) -> str:
        raise NotImplementedError
```

### Phase 2: Implement the Retrieval Tool
Wrap your existing `VectorStore` (from `rag_engine/vector_store`) into a tool.

**File:** `rag_engine/agentic_rag/tools.py`
```python
class LocalKnowledgeTool(BaseTool):
    def __init__(self, vector_store, embedder):
        super().__init__(
            name="search_knowledge_base",
            description="Useful for answering questions about [YOUR DOMAIN]. Input should be a specific search query."
        )
        self.vs = vector_store
        self.embedder = embedder

    def run(self, query: str) -> str:
        # Reuse existing logic from SimpleRAGEngine.retrieve
        query_vec = self.embedder.embed_text(query)
        results = self.vs.search(query_vec, top_k=3)
        # Format results as a string for the LLM
        return "\n".join([f"Content: {r[0]}" for r in results])
```

### Phase 3: Design the Agent Prompt
The prompt is the most critical part. It tells the LLM how to behave.

**File:** `rag_engine/agentic_rag/prompts.py`
```python
AGENT_SYSTEM_PROMPT = """
You are an intelligent research assistant. You have access to the following tools:

{tool_descriptions}

To use a tool, please use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

Begin!

Question: {input}
"""
```

### Phase 4: Implement the Agent Engine
Create the engine that drives the loop.

**File:** `rag_engine/agentic_rag/engine.py`
```python
class AgenticRAGEngine:
    def __init__(self, vector_store, embedder, llm_provider):
        self.tools = [LocalKnowledgeTool(vector_store, embedder)]
        self.llm = llm_provider # Wrapper for Ollama/OpenAI
        
    def chat(self, user_query):
        # 1. Prepare Prompt
        prompt = AGENT_SYSTEM_PROMPT.format(
            tool_descriptions=..., 
            tool_names=..., 
            input=user_query
        )
        
        # 2. Start Loop (Max steps: 5)
        messages = [{"role": "system", "content": prompt}]
        
        for _ in range(5):
            # Call LLM
            response = self.llm.chat(messages)
            
            # Parse for "Action:"
            if "Action:" in response and "Action Input:" in response:
                # Execute Tool
                tool_name = parse_action(response)
                tool_input = parse_input(response)
                
                observation = self.execute_tool(tool_name, tool_input)
                
                # Append result to history
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            
            elif "Final Answer:" in response:
                return parse_final_answer(response)
                
        return "I could not find an answer in the allotted steps."
```

### Phase 5: Integration
1.  Update `rag_engine/__init__.py` to export `AgenticRAGEngine`.
2.  Update your UI/API to allow selecting "Agentic" as a RAG mode.
3.  Pass the same `vector_store` and `embedder` instances to `AgenticRAGEngine` as you do for `SimpleRAGEngine`.

## 4. Key Considerations
*   **Latency**: Agentic RAG is slower because it involves multiple LLM calls (Reasoning -> Tool -> Reasoning -> Answer).
*   **Model Quality**: Small local models (like Llama-3-8b-q4) might struggle with following the strict "Thought/Action" format. You might need to use a simpler "Router" pattern for smaller models.
*   **Router Pattern Alternative**:
    *   Step 1: Ask LLM "Is this question about X (use vector db) or general chat?"
    *   Step 2: If X, run Vector Search. If General, just answer.
    *   This is faster and more robust for smaller models.
