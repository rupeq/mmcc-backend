from functools import lru_cache

from argon2 import PasswordHasher
from argon2.low_level import Type

from src.config import get_settings


@lru_cache()
def get_password_hasher() -> PasswordHasher:
    """Get a cached instance of the PasswordHasher."""
    settings = get_settings().argon_settings
    return PasswordHasher(
        time_cost=settings.argon_time_cost,
        memory_cost=settings.argon_memory_cost,
        parallelism=settings.argon_parallelism,
        hash_len=32,
        type=Type.ID,
    )


def hash_password(password: str) -> str:
    """
    Hash a given password using Argon2.

    Args:
        password (str): The plaintext password to hash.

    Returns:
        str: The Argon2 hashed password.
    """
    return get_password_hasher().hash(password)


def verify_password(password: str, *, password_hash: str) -> bool:
    """
    Verify a plaintext password against a stored Argon2 hash.

    Args:
        password (str): The plaintext password to verify.
        password_hash (str): The Argon2 hashed password from storage.

    Returns:
        bool: True if the password matches the hash, False otherwise.
    """
    try:
        return get_password_hasher().verify(password_hash, password)
    except Exception:
        return False
