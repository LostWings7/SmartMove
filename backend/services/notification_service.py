"""
SafeCity Loop V2 — Notification Service
Authoritative database-backed in-app notifications for citizens and municipal staff.
"""
from typing import List, Optional
from sqlalchemy.orm import Session
from backend.models import Notification


def create_notification(
    db: Session,
    user_id: int,
    type: str,
    title: str,
    message: str,
    complaint_id: Optional[int] = None,
) -> Notification:
    """Creates and persists an in-app notification for a specific user."""
    notif = Notification(
        user_id=user_id,
        type=type,
        title=title,
        message=message,
        related_complaint_id=complaint_id,
        is_read=False,
    )
    db.add(notif)
    db.flush()
    return notif


def get_user_notifications(
    db: Session,
    user_id: int,
    limit: int = 50,
) -> List[Notification]:
    """Retrieves recent notifications for the authenticated user."""
    return (
        db.query(Notification)
        .filter(Notification.user_id == user_id)
        .order_by(Notification.created_at.desc())
        .limit(limit)
        .all()
    )


def get_unread_count(db: Session, user_id: int) -> int:
    """Counts unread notifications for the user."""
    return (
        db.query(Notification)
        .filter(Notification.user_id == user_id, Notification.is_read == False)
        .count()
    )


def mark_notification_read(db: Session, notification_id: int, user_id: int) -> bool:
    """Marks a single notification as read if owned by user."""
    notif = (
        db.query(Notification)
        .filter(Notification.id == notification_id, Notification.user_id == user_id)
        .first()
    )
    if notif:
        notif.is_read = True
        db.commit()
        return True
    return False


def mark_all_notifications_read(db: Session, user_id: int) -> int:
    """Marks all unread notifications for user as read."""
    count = (
        db.query(Notification)
        .filter(Notification.user_id == user_id, Notification.is_read == False)
        .update({"is_read": True})
    )
    db.commit()
    return count
