"""
SafeCity Loop V2 — In-App Notifications Router
Database-backed notification center for citizens and municipal staff.
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User, Notification
from backend.schemas import NotificationOut, NotificationCountOut
from backend.auth import get_current_user
from backend.services.notification_service import (
    get_user_notifications, get_unread_count,
    mark_notification_read, mark_all_notifications_read
)

router = APIRouter(prefix="/notifications", tags=["Notifications"])


@router.get("", response_model=List[NotificationOut])
def get_my_notifications(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Returns authoritative notification list for the authenticated user."""
    notifs = get_user_notifications(db, current_user.id, limit=limit)
    return [NotificationOut.model_validate(n) for n in notifs]


@router.get("/unread-count", response_model=NotificationCountOut)
def get_my_unread_count(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Returns the number of unread notifications for notification badges."""
    count = get_unread_count(db, current_user.id)
    return NotificationCountOut(unread_count=count)


@router.post("/{notification_id}/read")
def mark_read(
    notification_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Marks a single notification as read."""
    success = mark_notification_read(db, notification_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found.")
    return {"status": "success", "message": "Notification marked as read."}


@router.post("/read-all")
def mark_all_read(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Marks all notifications for the authenticated user as read."""
    count = mark_all_notifications_read(db, current_user.id)
    return {"status": "success", "count": count, "message": f"{count} notifications marked as read."}
