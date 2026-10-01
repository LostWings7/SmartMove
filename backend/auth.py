"""
SafeCity Loop V2 — Authentication & Role-Based Access Control (RBAC)
Authoritative JWT token issuance, verification, bcrypt password hashing,
and strict endpoint-level permission enforcement.
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import jwt
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from backend.database import get_db
from backend.models import User, UserRole

load_dotenv()

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "safecity-loop-v2-auth-secret-key-prod-2026")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

# Bearer security scheme
security_bearer = HTTPBearer(auto_error=False)


# ── Password Hashing ──────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Hashes a plaintext password using bcrypt with salt."""
    if not password:
        raise ValueError("Password cannot be empty.")
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Securely verifies a plaintext password against a stored bcrypt hash."""
    if not plain_password or not hashed_password:
        return False
    try:
        pwd_bytes = plain_password.encode("utf-8")[:72]
        hash_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(pwd_bytes, hash_bytes)
    except Exception:
        return False


# ── JWT Token Utilities ───────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Encodes a signed JWT access token containing user claims."""
    to_encode = data.copy()
    now_utc = datetime.now(timezone.utc)
    if expires_delta:
        expire = now_utc + expires_delta
    else:
        expire = now_utc + timedelta(hours=JWT_EXPIRATION_HOURS)
    to_encode.update({"exp": expire, "iat": now_utc})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decodes and cryptographically validates a JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token signature or payload.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── FastAPI Dependencies for Auth & RBAC ──────────────────────────────────────

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
    db: Session = Depends(get_db)
) -> User:
    """Extracts, verifies Bearer token, and retrieves authoritative User entity."""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Bearer token missing.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token payload is missing subject identity.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User associated with this token no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account has been suspended or deactivated. Contact municipal administration.",
        )

    return user


def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Retrieves user if valid Bearer token provided, else None."""
    if not credentials or not credentials.credentials:
        return None
    try:
        payload = decode_access_token(credentials.credentials)
        user_id = payload.get("sub")
        if not user_id:
            return None
        user = db.query(User).filter(User.id == int(user_id), User.is_active == True).first()
        return user
    except Exception:
        return None


def require_role(allowed_roles: List[str]):
    """Returns a dependency checking that authenticated user belongs to allowed roles."""
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied: Requires one of roles {allowed_roles}. Current role: '{current_user.role}'.",
            )
        return current_user
    return role_checker


# Convenient role-based dependencies
require_citizen_user = require_role([UserRole.CITIZEN, UserRole.ADMIN, UserRole.SUPERUSER])
require_staff_user = require_role(UserRole.STAFF)
require_inspector_user = require_role([UserRole.INSPECTOR, UserRole.ADMIN, UserRole.SUPERUSER])
require_engineer_user = require_role([UserRole.ENGINEER, UserRole.ADMIN, UserRole.SUPERUSER])
require_admin_user = require_role([UserRole.ADMIN, UserRole.SUPERUSER])
require_superuser_only = require_role([UserRole.SUPERUSER])
