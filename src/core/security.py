from argon2 import PasswordHasher
from argon2.low_level import Type

from src.config import get_settings

ph = PasswordHasher(
    time_cost=get_settings().argon_settings.argon_time_cost,
    memory_cost=get_settings().argon_settings.argon_memory_cost,
    parallelism=get_settings().argon_settings.argon_parallelism,
    hash_len=32,
    type=Type.ID,
)


def hash_password(password: str) -> str:
    return ph.hash(password)


def verify_password(password: str, *, password_hash: str) -> bool:
    try:
        return ph.verify(password_hash, password)
    except Exception:
        return False
