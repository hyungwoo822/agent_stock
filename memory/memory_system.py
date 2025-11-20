import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from langchain.memory import ConversationBufferMemory
from config.settings import config
from memory.local_memory import LocalVectorStore, LocalEpisodeMemory

class UserProfileStore:
    """User investment profile storage (Local JSON)"""
    def __init__(self, filepath: str = "./data/user_profile.json"):
        self.filepath = filepath
        
    def update_profile(self, user_id: str, profile_data: Dict):
        """Update user profile"""
        profile = {
            "risk_tolerance": profile_data.get("risk_tolerance", "moderate"),
            "investment_horizon": profile_data.get("investment_horizon", "long_term"),
            "preferred_sectors": profile_data.get("preferred_sectors", []),
            "avoided_sectors": profile_data.get("avoided_sectors", []),
            "target_return": profile_data.get("target_return", 0.15),
            "max_drawdown": profile_data.get("max_drawdown", 0.20),
            "updated_at": datetime.now().isoformat()
        }
        
        with open(self.filepath, 'w') as f:
            json.dump(profile, f, indent=2)
    
    def get_profile(self, user_id: str) -> Dict:
        """Get user profile"""
        try:
            with open(self.filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "risk_tolerance": "moderate",
                "investment_horizon": "long_term",
                "preferred_sectors": [],
                "avoided_sectors": [],
                "target_return": 0.15,
                "max_drawdown": 0.20
            }

class HybridMemorySystem:
    """Integrated Memory System (Local)"""
    def __init__(self, user_id: str):
        self.user_id = user_id
        
        # Short-term Memory
        self.short_term = ConversationBufferMemory(
            memory_key="chat_history",
            max_token_limit=4000,
            return_messages=True
        )
        
        # Long-term Memory (Local ChromaDB)
        self.long_term = LocalVectorStore(collection_name=f"user_{user_id}_vectors")
        
        # Episode Memory (Local JSON)
        self.episode_memory = LocalEpisodeMemory()
        
        # User Profile
        self.user_profile = UserProfileStore()
        
    def add_interaction(self, message: str, response: str, metadata: Dict = None):
        """Add interaction to memory"""
        # Short-term
        self.short_term.save_context(
            {"input": message},
            {"output": response}
        )
        
        # Long-term (if important)
        if metadata and metadata.get("importance", 0) > 0.7:
            combined_text = f"User: {message}\nAssistant: {response}"
            self.long_term.add_documents(
                [combined_text],
                [metadata]
            )
    
    def search_relevant_memory(self, query: str, memory_type: str = "all") -> List[Dict]:
        """Search relevant memory"""
        results = []
        
        if memory_type in ["all", "short"]:
            # Short-term memory search
            recent_messages = self.short_term.chat_memory.messages[-10:]
            results.extend([
                {"type": "short_term", "content": msg.content}
                for msg in recent_messages
            ])
        
        if memory_type in ["all", "long"]:
            # Long-term memory search
            similar_docs = self.long_term.similarity_search(query, k=5)
            results.extend([
                {"type": "long_term", "content": doc[0], "score": doc[2]}
                for doc in similar_docs
            ])
        
        if memory_type in ["all", "episode"]:
            # Episode memory search (simplified for local)
            episodes = self.episode_memory.get_recent_episodes(limit=3)
            results.extend([
                {"type": "episode", "content": episode}
                for episode in episodes
            ])
        
        return results


