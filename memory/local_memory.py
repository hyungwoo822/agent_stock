import os
import json
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LocalVectorStore:
    def __init__(self, collection_name: str = "market_memory"):
        # Initialize ChromaDB in persistent mode
        self.client = chromadb.PersistentClient(path="./data/chroma_db")
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """Add documents to ChromaDB"""
        if not texts:
            return
            
        ids = [f"doc_{datetime.now().timestamp()}_{i}" for i in range(len(texts))]
        
        # Ensure metadatas is not None
        if metadatas is None:
            metadatas = [{}] * len(texts)
            
        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
    def similarity_search(self, query: str, k: int = 5) -> List[tuple]:
        """Search for similar documents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        
        # Format results to match expected output [(doc, score), ...]
        formatted_results = []
        if results['documents']:
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i] if results['metadatas'] else {}
                dist = results['distances'][0][i] if results['distances'] else 0
                
                # Create a mock document object to match LangChain interface if needed, 
                # or just return dict
                formatted_results.append((doc, meta, dist))
                
        return formatted_results

class LocalEpisodeMemory:
    def __init__(self, filepath: str = "./data/episodes.json"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self._load_episodes()
        
    def _load_episodes(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.episodes = json.load(f)
            except:
                self.episodes = []
        else:
            self.episodes = []
            
    def _save_episodes(self):
        with open(self.filepath, 'w') as f:
            json.dump(self.episodes, f, indent=2)
            
    def store_episode(self, episode_data: Dict):
        """Store a trading episode"""
        episode = {
            "id": f"ep_{datetime.now().timestamp()}",
            "timestamp": datetime.now().isoformat(),
            **episode_data
        }
        self.episodes.append(episode)
        self._save_episodes()
        
    def get_recent_episodes(self, limit: int = 5) -> List[Dict]:
        """Get most recent episodes"""
        return sorted(self.episodes, key=lambda x: x['timestamp'], reverse=True)[:limit]
