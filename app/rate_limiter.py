import os
from fastapi import FastAPI, Request, HTTPException
from functools import wraps
import time

class RateLimiter:
    """Simple in-memory rate limiter for serverless environment."""
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}

    def is_allowed(self, ip: str) -> bool:
        current_time = time.time()
        
        # Clean up old requests
        if ip in self.requests:
            self.requests[ip] = [t for t in self.requests[ip] if current_time - t < self.window_seconds]
        else:
            self.requests[ip] = []
            
        # Check limit
        if len(self.requests[ip]) >= self.max_requests:
            return False
            
        # Add new request
        self.requests[ip].append(current_time)
        return True

# Global instance
limiter = RateLimiter(max_requests=10, window_seconds=60)

def rate_limit(func):
    """Decorator for rate limiting endpoints."""
    @wraps(func)
    async def wrapper(request: Request, *args, **kwargs):
        # Extract IP, fallback to 127.0.0.1
        client_ip = request.client.host if request.client else "127.0.0.1"
        if request.headers.get("x-forwarded-for"):
            client_ip = request.headers.get("x-forwarded-for").split(",")[0].strip()
            
        if not limiter.is_allowed(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please try again later."
            )
            
        return await func(request, *args, **kwargs)
    return wrapper
