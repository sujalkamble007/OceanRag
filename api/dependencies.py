"""
api/dependencies.py — Shared FastAPI dependencies: JWT auth, role guards.
"""
import time
from fastapi import Depends, HTTPException, Header, status
from api.auth.utils import decode_token
from core.database import get_user_by_username


async def get_current_user(authorization: str = Header(...)) -> dict:
    """
    Extracts the Bearer JWT from the Authorization header,
    decodes it, fetches the user from DB, and returns user dict.
    Raises HTTP 401 if token is missing, invalid, or user not found.
    Retries DB lookups up to 3 times to handle Neon connection timeouts.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must start with 'Bearer '",
        )
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)

    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    # Retry DB lookup up to 3 times (Neon can intermittently timeout)
    last_err = None
    for attempt in range(3):
        try:
            user = get_user_by_username(username)
            if not user:
                raise HTTPException(status_code=401, detail="User not found")
            return user
        except HTTPException:
            raise  # Don't retry auth failures, only DB errors
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(1)  # Wait 1s before retrying

    raise HTTPException(
        status_code=503,
        detail=f"Database temporarily unavailable after 3 attempts: {str(last_err)}",
    )


def require_role(*allowed_roles: str):
    """
    Factory dependency — only passes if current user has one of the allowed roles.
    Usage:
        @router.get("/admin-only", dependencies=[Depends(require_role("admin"))])
        OR
        current_user = Depends(require_role("admin", "researcher"))
    """
    async def _check(current_user: dict = Depends(get_current_user)) -> dict:
        if current_user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {allowed_roles}",
            )
        return current_user
    return _check
