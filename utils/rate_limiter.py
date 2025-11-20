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