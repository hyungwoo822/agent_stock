import time
from enum import Enum
from typing import Callable, Any
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitState.HALF_OPEN
                else:
                    raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
    
    def _on_success(self):
        self.failure_count = 0
        self.state = CircuitState.CLOSED
    
    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

# utils/rate_limiter.py
import time
import redis
from typing import Optional
from functools import wraps
from config.settings import config

class RateLimiter:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            decode_responses=True
        )
    
    def limit(self, key: str, max_calls: int, period: int):
        """Decorator for rate limiting"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                pipe = self.redis_client.pipeline()
                now = time.time()
                pipeline_key = f"rate_limit:{key}:{func.__name__}"
                
                # Remove old entries
                pipe.zremrangebyscore(pipeline_key, 0, now - period)
                
                # Count current entries
                pipe.zcard(pipeline_key)
                
                # Add current request
                pipe.zadd(pipeline_key, {str(now): now})
                
                # Set expiry
                pipe.expire(pipeline_key, period)
                
                results = pipe.execute()
                
                if results[1] >= max_calls:
                    raise Exception(f"Rate limit exceeded for {key}")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator

# utils/encryption.py
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2

class EncryptionManager:
    def __init__(self, password: str):
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'portfolio_salt',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        self.cipher = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        return self.cipher.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        return self.cipher.decrypt(encrypted_data.encode()).decode()