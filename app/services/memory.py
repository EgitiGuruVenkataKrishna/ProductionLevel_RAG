import json
import logging
from typing import List, Dict
from app.db.redis_client import get_redis

logger = logging.getLogger(__name__)

# Max number of messages to keep in history
MAX_HISTORY_MESSAGES = 10  # 5 turns = 10 messages (user + assistant)
MEMORY_TTL = 86400  # 24 hours

async def add_message(session_id: str, role: str, content: str):
    """Add a message to the session's chat history in Redis."""
    try:
        r = await get_redis()
        key = f"memory:{session_id}"
        message = json.dumps({"role": role, "content": content})
        
        # RPUSH adds to the end of the list
        await r.rpush(key, message)
        
        # Trim list to keep only the latest MAX_HISTORY_MESSAGES
        await r.ltrim(key, -MAX_HISTORY_MESSAGES, -1)
        
        # Refresh TTL
        await r.expire(key, MEMORY_TTL)
    except Exception as e:
        logger.error(f"Error adding message to memory: {e}")

async def get_history(session_id: str) -> List[Dict[str, str]]:
    """Retrieve the recent chat history for a session."""
    try:
        r = await get_redis()
        key = f"memory:{session_id}"
        
        messages = await r.lrange(key, 0, -1)
        
        return [json.loads(msg) for msg in messages]
    except Exception as e:
        logger.error(f"Error retrieving memory: {e}")
        return []
