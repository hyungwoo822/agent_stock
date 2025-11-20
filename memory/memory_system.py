import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import boto3
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import redis
from config.settings import config

class S3VectorStore:
    def __init__(self, bucket: str, prefix: str = "vectors"):
        self.s3_client = boto3.client('s3', region_name=config.AWS_REGION)
        self.bucket = bucket
        self.prefix = prefix
        self.embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
        self.local_store = FAISS.from_texts(["initialization"], self.embeddings)
        
    def add_documents(self, texts: List[str], metadatas: List[Dict] = None):
        """증분 업데이트 - Delta만 추가"""
        # 임베딩 생성
        embeddings = self.embeddings.embed_documents(texts)
        
        # 로컬 FAISS에 추가
        self.local_store.add_texts(texts, metadatas)
        
        # S3에 저장 (증분 방식)
        timestamp = datetime.now().isoformat()
        delta_key = f"{self.prefix}/delta_{timestamp}.pkl"
        
        delta_data = {
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": metadatas,
            "timestamp": timestamp
        }
        
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=delta_key,
            Body=pickle.dumps(delta_data)
        )
        
        # 일정 크기 이상이면 컴팩션
        if len(self.local_store.docstore._dict) > 10000:
            self._compact_vectors()
    
    def _compact_vectors(self):
        """벡터 컴팩션 - 오래된 델타 병합"""
        # S3에서 모든 델타 파일 목록 가져오기
        response = self.s3_client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=f"{self.prefix}/delta_"
        )
        
        if 'Contents' not in response:
            return
        
        # 30일 이상 된 델타 파일들 병합
        cutoff_date = datetime.now() - timedelta(days=30)
        old_deltas = []
        
        for obj in response['Contents']:
            key_date = obj['Key'].split('delta_')[1].split('.')[0]
            if datetime.fromisoformat(key_date) < cutoff_date:
                old_deltas.append(obj['Key'])
        
        if len(old_deltas) > 10:  # 10개 이상일 때만 컴팩션
            self._merge_deltas(old_deltas)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """유사도 검색"""
        return self.local_store.similarity_search_with_score(query, k=k)

class EpisodeMemory:
    """특정 거래 에피소드 기억"""
    def __init__(self):
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            decode_responses=True
        )
        self.episode_key_prefix = "episode:"
    
    def store_episode(self, episode_id: str, episode_data: Dict):
        """거래 에피소드 저장"""
        key = f"{self.episode_key_prefix}{episode_id}"
        
        episode = {
            "timestamp": datetime.now().isoformat(),
            "decision": episode_data.get("decision"),
            "context": episode_data.get("context"),
            "outcome": episode_data.get("outcome"),
            "profit_loss": episode_data.get("profit_loss"),
            "lessons_learned": episode_data.get("lessons_learned")
        }
        
        self.redis_client.hset(key, mapping={
            k: json.dumps(v) if isinstance(v, dict) else v
            for k, v in episode.items()
        })
        
        # TTL 설정 (90일)
        self.redis_client.expire(key, 90 * 24 * 3600)
    
    def get_similar_episodes(self, context: Dict, limit: int = 5) -> List[Dict]:
        """유사한 과거 에피소드 검색"""
        # 모든 에피소드 키 가져오기
        pattern = f"{self.episode_key_prefix}*"
        episode_keys = self.redis_client.keys(pattern)
        
        episodes = []
        for key in episode_keys:
            episode_data = self.redis_client.hgetall(key)
            episode = {
                k: json.loads(v) if k in ['context', 'outcome'] else v
                for k, v in episode_data.items()
            }
            
            # 컨텍스트 유사도 계산 (간단한 버전)
            similarity = self._calculate_similarity(context, episode.get('context', {}))
            episode['similarity'] = similarity
            episodes.append(episode)
        
        # 유사도 순으로 정렬
        episodes.sort(key=lambda x: x['similarity'], reverse=True)
        return episodes[:limit]
    
    def _calculate_similarity(self, context1: Dict, context2: Dict) -> float:
        """간단한 유사도 계산"""
        if not context2:
            return 0.0
        
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        similarity = 0
        for key in common_keys:
            if context1[key] == context2[key]:
                similarity += 1
        
        return similarity / len(set(context1.keys()) | set(context2.keys()))

class UserProfileStore:
    """사용자 투자 성향 저장"""
    def __init__(self):
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            decode_responses=True
        )
    
    def update_profile(self, user_id: str, profile_data: Dict):
        """사용자 프로필 업데이트"""
        key = f"user_profile:{user_id}"
        
        profile = {
            "risk_tolerance": profile_data.get("risk_tolerance", "moderate"),
            "investment_horizon": profile_data.get("investment_horizon", "long_term"),
            "preferred_sectors": json.dumps(profile_data.get("preferred_sectors", [])),
            "avoided_sectors": json.dumps(profile_data.get("avoided_sectors", [])),
            "target_return": profile_data.get("target_return", 0.15),
            "max_drawdown": profile_data.get("max_drawdown", 0.20),
            "rebalancing_frequency": profile_data.get("rebalancing_frequency", "monthly"),
            "tax_strategy": profile_data.get("tax_strategy", "standard"),
            "updated_at": datetime.now().isoformat()
        }
        
        self.redis_client.hset(key, mapping=profile)
    
    def get_profile(self, user_id: str) -> Dict:
        """사용자 프로필 조회"""
        key = f"user_profile:{user_id}"
        profile_data = self.redis_client.hgetall(key)
        
        if not profile_data:
            # 기본 프로필 생성
            default_profile = {
                "risk_tolerance": "moderate",
                "investment_horizon": "long_term",
                "preferred_sectors": [],
                "avoided_sectors": [],
                "target_return": 0.15,
                "max_drawdown": 0.20
            }
            self.update_profile(user_id, default_profile)
            return default_profile
        
        # JSON 필드 파싱
        profile_data['preferred_sectors'] = json.loads(profile_data.get('preferred_sectors', '[]'))
        profile_data['avoided_sectors'] = json.loads(profile_data.get('avoided_sectors', '[]'))
        profile_data['target_return'] = float(profile_data.get('target_return', 0.15))
        profile_data['max_drawdown'] = float(profile_data.get('max_drawdown', 0.20))
        
        return profile_data

class HybridMemorySystem:
    """통합 메모리 시스템"""
    def __init__(self, user_id: str):
        self.user_id = user_id
        
        # Short-term Memory
        self.short_term = ConversationBufferMemory(
            memory_key="chat_history",
            max_token_limit=4000,
            return_messages=True
        )
        
        # Long-term Memory (S3 Vectors)
        self.long_term = S3VectorStore(
            bucket=config.S3_BUCKET,
            prefix=f"users/{user_id}/vectors"
        )
        
        # Episode Memory
        self.episode_memory = EpisodeMemory()
        
        # User Profile
        self.user_profile = UserProfileStore()
        
    def add_interaction(self, message: str, response: str, metadata: Dict = None):
        """상호작용 추가"""
        # Short-term에 추가
        self.short_term.save_context(
            {"input": message},
            {"output": response}
        )
        
        # Long-term에 추가 (중요한 정보만)
        if metadata and metadata.get("importance", 0) > 0.7:
            combined_text = f"User: {message}\nAssistant: {response}"
            self.long_term.add_documents(
                [combined_text],
                [metadata]
            )
    
    def search_relevant_memory(self, query: str, memory_type: str = "all") -> List[Dict]:
        """관련 메모리 검색"""
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
                {"type": "long_term", "content": doc[0].page_content, "score": doc[1]}
                for doc in similar_docs
            ])
        
        if memory_type in ["all", "episode"]:
            # Episode memory search
            context = {"query": query}
            episodes = self.episode_memory.get_similar_episodes(context, limit=3)
            results.extend([
                {"type": "episode", "content": episode}
                for episode in episodes
            ])
        
        return results

# memory/session_manager.py
from typing import Dict, Optional
from datetime import datetime, timedelta
import json
import uuid
import redis
from config.settings import config

class SessionManager:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            decode_responses=True
        )
        self.session_ttl = 24 * 3600  # 24 hours
    
    def create_session(self, user_id: str) -> str:
        """새 세션 생성"""
        session_id = str(uuid.uuid4())
        session_key = f"session:{session_id}"
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_active": datetime.now().isoformat(),
            "state": json.dumps({
                "market_condition": None,
                "portfolio_state": {},
                "pending_decisions": [],
                "current_workflow": None,
                "workflow_state": {}
            })
        }
        
        self.redis_client.hset(session_key, mapping=session_data)
        self.redis_client.expire(session_key, self.session_ttl)
        
        # 사용자별 세션 목록에 추가
        user_sessions_key = f"user_sessions:{user_id}"
        self.redis_client.sadd(user_sessions_key, session_id)
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """세션 조회"""
        session_key = f"session:{session_id}"
        session_data = self.redis_client.hgetall(session_key)
        
        if not session_data:
            return None
        
        # JSON 필드 파싱
        session_data['state'] = json.loads(session_data.get('state', '{}'))
        
        # 마지막 활동 시간 업데이트
        self.redis_client.hset(session_key, "last_active", datetime.now().isoformat())
        self.redis_client.expire(session_key, self.session_ttl)
        
        return session_data
    
    def update_session_state(self, session_id: str, state_updates: Dict):
        """세션 상태 업데이트"""
        session_key = f"session:{session_id}"
        
        # 현재 상태 가져오기
        current_state = self.redis_client.hget(session_key, "state")
        if current_state:
            current_state = json.loads(current_state)
        else:
            current_state = {}
        
        # 상태 업데이트
        current_state.update(state_updates)
        
        # 저장
        self.redis_client.hset(session_key, "state", json.dumps(current_state))
        self.redis_client.hset(session_key, "last_active", datetime.now().isoformat())
        self.redis_client.expire(session_key, self.session_ttl)
    
    def get_user_sessions(self, user_id: str) -> List[str]:
        """사용자의 모든 세션 조회"""
        user_sessions_key = f"user_sessions:{user_id}"
        return list(self.redis_client.smembers(user_sessions_key))
    
    def cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        # Redis TTL이 자동으로 처리하므로 추가 작업 불필요
        pass