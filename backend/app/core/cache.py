"""
Redis Cache Management

Provides async Redis client for caching and pub/sub.
"""

from typing import Any, Optional
import json
import redis.asyncio as redis
from redis.asyncio import Redis

from app.config import settings

# Global Redis client
_redis_client: Redis | None = None


async def init_redis() -> None:
    """Initialize Redis connection (optional for local dev)."""
    global _redis_client
    try:
        _redis_client = redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        # Test connection
        await _redis_client.ping()
        print("✅ Redis connected successfully")
    except Exception as e:
        print(f"⚠️  Redis not available ({e}). Running without cache.")
        _redis_client = None  # Allow app to run without Redis


async def close_redis() -> None:
    """Close Redis connection."""
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None


def get_redis() -> Redis | None:
    """Get Redis client instance (may be None if Redis not available)."""
    return _redis_client


class CacheService:
    """
    High-level caching service with JSON serialization.
    """
    
    def __init__(self, prefix: str = "adclass"):
        self.prefix = prefix
        self.default_ttl = settings.REDIS_CACHE_TTL
    
    def _make_key(self, key: str) -> str:
        """Create namespaced cache key."""
        return f"{self.prefix}:{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache, returns None if not found."""
        client = get_redis()
        value = await client.get(self._make_key(key))
        if value is not None:
            return json.loads(value)
        return None
    
    async def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with optional TTL."""
        client = get_redis()
        return await client.setex(
            self._make_key(key),
            ttl or self.default_ttl,
            json.dumps(value, default=str)
        )
    
    async def delete(self, key: str) -> int:
        """Delete key from cache."""
        client = get_redis()
        return await client.delete(self._make_key(key))
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        client = get_redis()
        return await client.exists(self._make_key(key)) > 0
    
    async def get_or_set(
        self, 
        key: str, 
        factory, 
        ttl: Optional[int] = None
    ) -> Any:
        """
        Get from cache or compute and cache if missing.
        
        Args:
            key: Cache key
            factory: Async callable to compute value if not cached
            ttl: Optional TTL override
        """
        value = await self.get(key)
        if value is not None:
            return value
        
        value = await factory()
        await self.set(key, value, ttl)
        return value
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        client = get_redis()
        cursor = 0
        deleted = 0
        
        while True:
            cursor, keys = await client.scan(
                cursor, 
                match=self._make_key(pattern),
                count=100
            )
            if keys:
                deleted += await client.delete(*keys)
            if cursor == 0:
                break
        
        return deleted


# Convenience instance
cache = CacheService()


# Specialized caches
predictions_cache = CacheService(prefix="adclass:predictions")
metrics_cache = CacheService(prefix="adclass:metrics")
client_cache = CacheService(prefix="adclass:clients")
