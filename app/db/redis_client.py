import logging
from redis.asyncio import Redis, ConnectionPool
from typing import Optional
from app.config import REDIS_URL

logger = logging.getLogger(__name__)

# Global connection pool
_redis_pool: Optional[ConnectionPool] = None

async def init_redis_pool() -> ConnectionPool:
    """Initialize and return the Redis connection pool."""
    global _redis_pool
    if _redis_pool is None:
        logger.info(f"Initializing Redis connection pool at {REDIS_URL}")
        _redis_pool = ConnectionPool.from_url(
            REDIS_URL,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            health_check_interval=30
        )
    return _redis_pool

async def get_redis() -> Redis:
    """
    Get an async Redis client instance.
    Uses connection pooling for efficiency.
    Returns a client that degrades gracefully (operations might raise exceptions if Redis is down).
    Calling code should handle redis.exceptions.ConnectionError gracefully.
    """
    pool = await init_redis_pool()
    return Redis(connection_pool=pool)

async def close_redis_pool():
    """Close the Redis connection pool during shutdown."""
    global _redis_pool
    if _redis_pool is not None:
        logger.info("Closing Redis connection pool")
        await _redis_pool.disconnect()
        _redis_pool = None
