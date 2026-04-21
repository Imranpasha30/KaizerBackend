"""Symmetric encryption for OAuth refresh/access tokens.

Single Fernet key sourced from settings.encryption_key (auto-provisioned to
.env on first boot). Rotation strategy when we outgrow a single key:
persist a key id alongside the ciphertext and expand to MultiFernet — not
needed for v1.
"""
from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken

from config import settings


class CryptoError(Exception):
    """Encryption or decryption failure."""


def _fernet() -> Fernet:
    key = settings.encryption_key
    if not key:
        raise CryptoError("KAIZER_ENCRYPTION_KEY is not set")
    try:
        return Fernet(key.encode("ascii") if isinstance(key, str) else key)
    except ValueError as e:
        raise CryptoError(f"Invalid Fernet key: {e}") from e


def encrypt(plaintext: str) -> str:
    """Return ciphertext as a base64 ASCII string safe to store in Text columns."""
    if not plaintext:
        return ""
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt(ciphertext: str) -> str:
    """Reverse of encrypt(). Raises CryptoError on malformed/tampered input."""
    if not ciphertext:
        return ""
    try:
        return _fernet().decrypt(ciphertext.encode("ascii")).decode("utf-8")
    except InvalidToken as e:
        raise CryptoError("Invalid or tampered ciphertext") from e
