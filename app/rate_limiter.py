import time
import logging
from fastapi import Request, HTTPException
from functools import wraps
from app.db.redis_client import get_redis
from app.config import RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS
import redis.exceptions

logger = logging.getLogger(__name__)

async def is_allowed(ip: str) -> bool:
    """Check if the given IP is allowed to make a request using Redis."""
    try:
        r = await get_redis()
        # Fixed window: Use the current minute (or window) as part of the key
        window_start = int(time.time() // RATE_LIMIT_WINDOW_SECONDS)
        key = f"ratelimit:{ip}:{window_start}"
        
        # Increment request count
        current_requests = await r.incr(key)
        
        # Set expiry on the key if it's the first request in this window
        if current_requests == 1:
            await r.expire(key, RATE_LIMIT_WINDOW_SECONDS + 5)  # Add 5 seconds buffer
            
        if current_requests > RATE_LIMIT_MAX_REQUESTS:
            return False
            
        return True
    except redis.exceptions.RedisError as e:
        # Graceful degradation: if Redis is down, allow the request but log a warning
        logger.warning(f"Redis rate limiter unavailable: {e}. Allowing request from {ip}.")
        return True
    except Exception as e:
        logger.error(f"Unexpected error in rate limiter: {e}")
        return True

def rate_limit(func):
    """Decorator for rate limiting endpoints."""
    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        # Extract IP, fallback to 127.0.0.1
        client_ip = request.client.host if request.client else "127.0.0.1"
        if request.headers.get("x-forwarded-for"):
            client_ip = request.headers.get("x-forwarded-for").split(",")[0].strip()
            
        allowed = await is_allowed(client_ip)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later."
            )
            
        return await func(request, *args, **kwargs)
    return wrapper
