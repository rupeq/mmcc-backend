from unittest.mock import patch
from argon2.exceptions import VerifyMismatchError

from src.core.security import hash_password, verify_password

TEST_PASSWORD = "mysecretpassword"


def test_hash_password():
    """
    Test that hash_password returns a non-empty string different from the original.
    """
    hashed = hash_password(TEST_PASSWORD)
    assert isinstance(hashed, str)
    assert hashed != TEST_PASSWORD
    assert len(hashed) > 0


def test_verify_password_success():
    """
    Test that a correctly hashed password verifies successfully.
    """
    hashed = hash_password(TEST_PASSWORD)
    assert verify_password(TEST_PASSWORD, password_hash=hashed) is True


def test_verify_password_failure_wrong_password():
    """
    Test that an incorrect password fails verification.
    """
    hashed = hash_password(TEST_PASSWORD)
    assert verify_password("wrongpassword", password_hash=hashed) is False


def test_verify_password_failure_invalid_hash():
    """
    Test that an invalid or malformed hash fails verification gracefully.
    """
    assert verify_password(TEST_PASSWORD, password_hash="invalidhash") is False


@patch("src.core.security.ph")
def test_verify_password_handles_argon_exception(mock_ph):
    """
    Test that any unexpected exception from the hashing library is handled.
    """
    mock_ph.verify.side_effect = VerifyMismatchError
    result = verify_password("anypassword", password_hash="anyhash")
    assert result is False
