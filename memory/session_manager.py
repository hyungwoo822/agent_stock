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