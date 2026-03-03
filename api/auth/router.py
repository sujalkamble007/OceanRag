"""
api/auth/router.py — Authentication endpoints: register, login, me.
"""
from fastapi import APIRouter, HTTPException, Depends, status
from api.auth.schemas import RegisterRequest, LoginRequest, TokenResponse, UserResponse
from api.auth.utils import hash_password, verify_password, create_access_token
from core.database import create_user, get_user_by_email, get_user_by_username
from api.dependencies import get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])

ALLOWED_ROLES = {"admin", "researcher", "student", "common_user"}


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(req: RegisterRequest):
    """Register a new user."""
    # Validate role
    if req.role not in ALLOWED_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role. Choose from: {ALLOWED_ROLES}")

    # Check duplicates
    if get_user_by_email(req.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    if get_user_by_username(req.username):
        raise HTTPException(status_code=400, detail="Username already taken")

    hashed = hash_password(req.password)
    user = create_user(req.username, req.email, hashed, req.role)
    return UserResponse(
        id=user["id"],
        username=user["username"],
        email=user["email"],
        role=user["role"],
    )


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    """Authenticate and return a JWT token."""
    user = get_user_by_email(req.email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if user.get("is_active") == "false":
        raise HTTPException(status_code=403, detail="Account is disabled")

    token = create_access_token({"sub": user["username"], "role": user["role"]})
    return TokenResponse(
        access_token=token,
        role=user["role"],
        username=user["username"],
    )


@router.get("/me", response_model=UserResponse)
def get_me(current_user: dict = Depends(get_current_user)):
    """Returns the currently logged-in user's profile."""
    return UserResponse(
        id=current_user["id"],
        username=current_user["username"],
        email=current_user["email"],
        role=current_user["role"],
    )
