import jwt
import logging
from fastapi import HTTPException, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import JWT_SECRET, JWT_ALGORITHM

logger = logging.getLogger(__name__)
security = HTTPBearer()

def verify_jwt(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    """
    Verify the JWT token from the Authorization header.
    Returns the decoded payload if valid, raises HTTPException otherwise.
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Expired JWT token")
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"JWT verification error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")

def get_current_user_session(request: Request, payload: dict = Security(verify_jwt)) -> str:
    """
    Extract the session ID from the JWT payload.
    Provides a secure session_id for the chat history, ignoring any client-provided session_id.
    """
    # The JWT could contain 'sub' or 'user_id' or 'session_id'
    # We will just use 'sub' as the session identifier for simplicity
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Invalid token payload: missing 'sub'")
    return sub
