import os
import json
import uuid
import shutil
from datetime import datetime

CHAT_DIR = os.path.join("data", "chat_history")
os.makedirs(CHAT_DIR, exist_ok=True)

class HistoryManager:
    def __init__(self):
        pass

    def create_new_chat(self, name="New Chat", doc_id=None):
        chat_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        chat_data = {
            "id": chat_id,
            "name": name,
            "doc_id": doc_id,
            "created_at": timestamp,
            "updated_at": timestamp,
            "messages": []
        }
        self.save_chat(chat_data)
        return chat_data

    def save_chat(self, chat_data):
        chat_id = chat_data["id"]
        chat_data["updated_at"] = datetime.now().isoformat()
        path = os.path.join(CHAT_DIR, f"{chat_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chat_data, f, indent=2)

    def load_chat(self, chat_id):
        path = os.path.join(CHAT_DIR, f"{chat_id}.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading chat {chat_id}: {e}")
            return None

    def list_chats(self):
        chats = []
        for f in os.listdir(CHAT_DIR):
            if f.endswith(".json"):
                try:
                    path = os.path.join(CHAT_DIR, f)
                    with open(path, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        chats.append({
                            "id": data.get("id"),
                            "name": data.get("name", "Untitled"),
                            "updated_at": data.get("updated_at", "")
                        })
                except Exception:
                    continue
        # Sort by updated_at desc
        chats.sort(key=lambda x: x["updated_at"], reverse=True)
        return chats

    def delete_chat(self, chat_id):
        path = os.path.join(CHAT_DIR, f"{chat_id}.json")
        if os.path.exists(path):
            os.remove(path)

    def rename_chat(self, chat_id, new_name):
        chat = self.load_chat(chat_id)
        if chat:
            chat["name"] = new_name
            self.save_chat(chat)
