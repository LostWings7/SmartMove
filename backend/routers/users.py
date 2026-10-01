"""
SafeCity Loop V2 — Admin User Management Router
Provides search, filtering, role assignment, active status toggle,
tamper-evident audit logging, and strict last-superuser safety protection.
"""
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import or_

from backend.database import get_db
from backend.models import User, UserRole, AuditLog, Complaint
from backend.schemas import (
    UserOut, UserRoleUpdateRequest, UserStatusUpdateRequest
)
from backend.auth import require_admin_user, get_current_user

router = APIRouter(prefix="/admin/users", tags=["Admin User Management"])


def _check_last_superuser_safety(target_user: User, db: Session, will_be_active: bool, new_role: Optional[str] = None):
    """
    Prevents the system from having zero active superusers.
    Raises HTTPException 400 if action would leave system without an active superuser.
    """
    if target_user.role == UserRole.SUPERUSER and target_user.is_active:
        is_demoting = new_role is not None and new_role != UserRole.SUPERUSER
        is_deactivating = not will_be_active
        if is_demoting or is_deactivating:
            active_superusers = (
                db.query(User)
                .filter(User.role == UserRole.SUPERUSER, User.is_active == True)
                .count()
            )
            if active_superusers <= 1:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Operation rejected: Cannot deactivate, delete, or demote the last active superuser.",
                )


@router.get("")
def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    role: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(None),
    search: Optional[str] = Query(None),
    current_user: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    """Lists users with filtering, search, and pagination."""
    q = db.query(User)

    if role:
        q = q.filter(User.role == role.upper())
    if is_active is not None:
        q = q.filter(User.is_active == is_active)
    if search:
        term = f"%{search.strip()}%"
        q = q.filter(or_(User.full_name.ilike(term), User.email.ilike(term), User.phone_number.ilike(term)))

    total = q.count()
    offset = (page - 1) * page_size
    users = q.order_by(User.created_at.desc()).offset(offset).limit(page_size).all()

    return {
        "items": [UserOut.model_validate(u) for u in users],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
    }


@router.get("/{user_id}", response_model=UserOut)
def get_user_detail(
    user_id: int,
    current_user: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    """Retrieves full user detail for administrators."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return UserOut.model_validate(user)


@router.patch("/{user_id}/role", response_model=UserOut)
def update_user_role(
    user_id: int,
    payload: UserRoleUpdateRequest,
    current_user: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    """
    Updates user role with last-superuser safety checks.
    Only existing SUPERUSER can promote or demote other SUPERUSER accounts.
    """
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found.")

    target_role = payload.role.upper()
    if target_role not in UserRole.ALL:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid role '{payload.role}'. Must be one of {UserRole.ALL}.",
        )

    # Superuser privilege check
    if (target_role == UserRole.SUPERUSER or target_user.role == UserRole.SUPERUSER) and current_user.role != UserRole.SUPERUSER:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only a SUPERUSER can modify or grant SUPERUSER privileges.",
        )

    # Enforce last-superuser safety
    _check_last_superuser_safety(target_user, db, will_be_active=target_user.is_active, new_role=target_role)

    old_role = target_user.role
    target_user.role = target_role
    target_user.updated_at = datetime.utcnow()

    db.add(AuditLog(
        actor=f"{current_user.full_name} ({current_user.role})",
        action="USER_ROLE_CHANGED",
        target_type="User",
        target_id=target_user.id,
        details=f"User {target_user.email} role changed from '{old_role}' to '{target_role}'.",
        timestamp=datetime.utcnow(),
    ))
    db.commit()
    db.refresh(target_user)
    return UserOut.model_validate(target_user)


@router.patch("/{user_id}/status", response_model=UserOut)
def update_user_status(
    user_id: int,
    payload: UserStatusUpdateRequest,
    current_user: User = Depends(require_admin_user),
    db: Session = Depends(get_db),
):
    """Activates or deactivates an account with last-superuser safety."""
    target_user = db.query(User).filter(User.id == user_id).first()
    if not target_user:
        raise HTTPException(status_code=404, detail="User not found.")

    if not payload.is_active:
        _check_last_superuser_safety(target_user, db, will_be_active=False)

    old_status = target_user.is_active
    target_user.is_active = payload.is_active
    target_user.updated_at = datetime.utcnow()

    db.add(AuditLog(
        actor=f"{current_user.full_name} ({current_user.role})",
        action="USER_STATUS_CHANGED",
        target_type="User",
        target_id=target_user.id,
        details=f"User {target_user.email} active status changed from '{old_status}' to '{payload.is_active}'.",
        timestamp=datetime.utcnow(),
    ))
    db.commit()
    db.refresh(target_user)
    return UserOut.model_validate(target_user)
