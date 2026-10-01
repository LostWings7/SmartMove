"""
SafeCity Loop V2 — Authentication & User Profile Router
Provides secure registration, bcrypt-verified login, profile management,
password changing, and tamper-evident audit trail integration.
"""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, UserRole, AuditLog
from backend.schemas import (
    UserRegisterRequest, UserLoginRequest, TokenResponse,
    UserOut, UserProfileUpdate, ChangePasswordRequest
)
from backend.auth import (
    hash_password, verify_password, create_access_token,
    get_current_user
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: UserRegisterRequest, db: Session = Depends(get_db)):
    """
    Registers a new citizen account with secure password hashing.
    Enforces email normalization, uniqueness, and creates audit log entry.
    """
    normalized_email = payload.email.strip().lower()
    if not normalized_email or "@" not in normalized_email:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="A valid email address is required.",
        )

    if len(payload.password) < 8:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Password must be at least 8 characters in length.",
        )

    # Check for existing email
    existing_user = db.query(User).filter(User.email == normalized_email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email address already exists.",
        )

    # Hash password
    pwd_hash = hash_password(payload.password)

    new_user = User(
        full_name=payload.full_name.strip(),
        email=normalized_email,
        phone_number=payload.phone_number.strip() if payload.phone_number else None,
        password_hash=pwd_hash,
        role=UserRole.CITIZEN,
        is_active=True,
        is_verified=True,
        preferred_language=payload.preferred_language or "en",
        last_login=datetime.utcnow(),
    )
    db.add(new_user)
    db.flush()

    # Log tamper-evident audit event
    db.add(AuditLog(
        actor=new_user.full_name,
        action="USER_REGISTERED",
        target_type="User",
        target_id=new_user.id,
        details=f"Citizen account registered with email: {normalized_email}",
        timestamp=datetime.utcnow(),
    ))
    db.commit()
    db.refresh(new_user)

    # Generate JWT
    token = create_access_token({"sub": str(new_user.id), "role": new_user.role, "email": new_user.email})

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserOut.model_validate(new_user),
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: UserLoginRequest, db: Session = Depends(get_db)):
    """
    Authoritative login endpoint. Verifies email & bcrypt hash,
    records last login, and produces signed JWT access token.
    """
    normalized_email = payload.email.strip().lower()
    user = db.query(User).filter(User.email == normalized_email).first()

    if not user or not verify_password(payload.password, user.password_hash):
        # Tamper-evident failed login audit
        db.add(AuditLog(
            actor=normalized_email,
            action="LOGIN_FAILED",
            target_type="User",
            target_id=user.id if user else None,
            details=f"Invalid login credentials attempted for: {normalized_email}",
            timestamp=datetime.utcnow(),
        ))
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Contact municipal administrator.",
        )

    # Update last login
    user.last_login = datetime.utcnow()
    db.add(AuditLog(
        actor=f"{user.full_name} ({user.role})",
        action="LOGIN_SUCCESS",
        target_type="User",
        target_id=user.id,
        details=f"Successful login for role: {user.role}",
        timestamp=datetime.utcnow(),
    ))
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": str(user.id), "role": user.role, "email": user.email})

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user=UserOut.model_validate(user),
    )


@router.get("/me", response_model=UserOut)
def get_current_user_profile(current_user: User = Depends(get_current_user)):
    """Returns the authenticated user's authoritative profile."""
    return UserOut.model_validate(current_user)


@router.patch("/profile", response_model=UserOut)
def update_profile(
    payload: UserProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Allows user to update allowable profile fields (name, phone, language, preferences)."""
    if payload.full_name is not None and payload.full_name.strip():
        current_user.full_name = payload.full_name.strip()
    if payload.phone_number is not None:
        current_user.phone_number = payload.phone_number.strip()
    if payload.preferred_language is not None:
        current_user.preferred_language = payload.preferred_language
    if payload.notification_preferences is not None:
        current_user.notification_preferences = payload.notification_preferences

    current_user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(current_user)
    return UserOut.model_validate(current_user)


@router.post("/change-password")
def change_password(
    payload: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Requires current password verification before updating bcrypt password hash."""
    if not verify_password(payload.current_password, current_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password verification failed.",
        )

    if len(payload.new_password) < 8:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="New password must be at least 8 characters in length.",
        )

    current_user.password_hash = hash_password(payload.new_password)
    current_user.updated_at = datetime.utcnow()

    db.add(AuditLog(
        actor=f"{current_user.full_name} ({current_user.role})",
        action="PASSWORD_CHANGED",
        target_type="User",
        target_id=current_user.id,
        details="User password successfully updated.",
        timestamp=datetime.utcnow(),
    ))
    db.commit()
    return {"status": "success", "message": "Password updated successfully."}


@router.post("/logout")
def logout(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Logs session logout event."""
    db.add(AuditLog(
        actor=f"{current_user.full_name} ({current_user.role})",
        action="LOGOUT",
        target_type="User",
        target_id=current_user.id,
        details="User logged out.",
        timestamp=datetime.utcnow(),
    ))
    db.commit()
    return {"status": "success", "message": "Successfully logged out."}
