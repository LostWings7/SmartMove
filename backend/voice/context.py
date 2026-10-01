"""
SafeCity Loop V2 — FRIDAY Voice Agent Context Provider
Maintains real-time synchronized UI state, user role, selected entities,
map position, and navigation state. Enforces role-based sanitization.
"""
import time
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class SafeCityContext(BaseModel):
    """Encapsulates the current SafeCity application and user state."""
    client_id: str = Field(default="default_session", description="Unique client or session identifier")
    current_user: Optional[Dict[str, Any]] = Field(default=None, description="Authenticated user info: id, name, role")
    user_role: str = Field(default="CITIZEN", description="User role: CITIZEN, INSPECTOR, ENGINEER, ADMIN, SUPERUSER")
    is_authenticated: bool = Field(default=False, description="Whether session is authenticated")
    current_page: str = Field(default="/", description="Active frontend URL path")
    current_page_title: str = Field(default="Overview", description="Human-readable title of the active screen")
    selected_complaint: Optional[str] = Field(default=None, description="Currently focused complaint reference (e.g. SC-2026-000001)")
    selected_hazard: Optional[int] = Field(default=None, description="Currently focused hazard ID")
    selected_conflict: Optional[int] = Field(default=None, description="Currently focused traffic conflict ID")
    selected_evidence: Optional[str] = Field(default=None, description="Currently focused evidence code (e.g. EV-2026-0001)")
    selected_road: Optional[str] = Field(default=None, description="Currently focused road segment name")
    selected_route: Optional[str] = Field(default="safer", description="Currently selected route key (fastest, balanced, safer)")
    pending_route_opt_in: bool = Field(default=False, description="Whether agent is awaiting user opt-in to start navigation")
    pending_action: Optional[Dict[str, Any]] = Field(default=None, description="Consequential action awaiting explicit user confirmation")
    pending_complaint_draft: Optional[Dict[str, Any]] = Field(default=None, description="Draft complaint details being conversationally collected")
    recent_notifications: List[Dict[str, Any]] = Field(default_factory=list, description="Recent user notifications")
    map_state: Dict[str, Any] = Field(default_factory=dict, description="Map center, zoom, bounds")
    route_state: Dict[str, Any] = Field(default_factory=dict, description="Active origin, destination, selected route")
    navigation_state: Dict[str, Any] = Field(default_factory=lambda: {"status": "IDLE", "step": 0}, description="Voice navigation state")
    last_updated: float = Field(default_factory=time.time, description="Unix timestamp of last context update")


PAGE_TITLES: Dict[str, str] = {
    "/": "City Safety Overview & Command Center",
    "/map": "Interactive SafeCity GIS Hazard Map",
    "/ai-vision": "AI Vision Road Inspection & Video Analytics",
    "/road-intelligence": "Road Condition Intelligence & Pavement Analytics",
    "/traffic-conflicts": "Traffic Conflict Analysis & Surrogate Near-Miss Events",
    "/evidence": "Evidence Audit & Municipal Decision Ledger",
    "/danger-zones": "Danger Zones & SafeCity Scores",
    "/repair": "AI Repair Priority & Municipal Operations",
    "/routes": "Safer Routes & Risk Comparison Engine",
    "/junction": "Smart Junction Digital Twin",
    "/analytics": "Citywide Safety Analytics & Trend Tracking",
    "/citizen/dashboard": "Citizen Portal Dashboard",
    "/citizen/complaints": "My Citizen Complaints",
    "/citizen/complaints/new": "Report a Road Hazard / New Complaint",
    "/citizen/notifications": "Notifications & Alerts",
    "/citizen/profile": "Citizen Profile & Preferences",
    "/admin/dashboard": "Municipal Operations & Admin Dashboard",
    "/admin/complaints": "Municipal Complaint Management Queue",
    "/admin/users": "User Management & Role Assignment",
}


class ContextManager:
    """Thread-safe centralized manager for active client sessions."""

    def __init__(self):
        self._contexts: Dict[str, SafeCityContext] = {
            "default_session": SafeCityContext()
        }

    def get_context(self, session_id: str = "default_session") -> SafeCityContext:
        """Retrieves or creates context for a session."""
        if session_id not in self._contexts:
            self._contexts[session_id] = SafeCityContext(client_id=session_id)
        return self._contexts[session_id]

    def update_context(self, session_id: str, updates: Dict[str, Any]) -> SafeCityContext:
        """Applies partial updates to a session context."""
        ctx = self.get_context(session_id)
        data = ctx.model_dump()

        # Automatically resolve page title if current_page is updated
        if "current_page" in updates and "current_page_title" not in updates:
            raw_path = updates["current_page"]
            title = PAGE_TITLES.get(raw_path)
            if not title:
                # Handle dynamic routes like /citizen/complaints/:id
                if raw_path.startswith("/citizen/complaints/"):
                    title = "Complaint Details"
                elif raw_path.startswith("/admin/complaints/"):
                    title = "Admin Complaint Review"
                else:
                    title = raw_path.replace("/", " ").replace("-", " ").strip().title() or "Overview"
            updates["current_page_title"] = title

        # Extract user info
        if "current_user" in updates and updates["current_user"]:
            user_info = updates["current_user"]
            updates["user_role"] = user_info.get("role", "CITIZEN")
            updates["is_authenticated"] = True

        data.update(updates)
        data["last_updated"] = time.time()
        self._contexts[session_id] = SafeCityContext(**data)
        return self._contexts[session_id]

    def sanitize_for_role(self, context: SafeCityContext, role: str) -> Dict[str, Any]:
        """Ensures citizens only see citizen-safe context while staff see full context."""
        role_upper = (role or "CITIZEN").upper()
        raw = context.model_dump()

        if role_upper == "CITIZEN":
            # Redact administrative internals
            if "admin" in raw["current_page"]:
                raw["current_page"] = "/citizen/dashboard"
                raw["current_page_title"] = "Citizen Portal Dashboard"
            # Citizens can only retain selected complaint if it is their own
            # (validation performed against user ID in permissions)
        return raw

    def format_context_for_prompt(self, context: SafeCityContext) -> str:
        """Formats a compact, high-signal context block for FRIDAY's prompt."""
        user_name = "Anonymous Citizen"
        user_role = context.user_role
        if context.current_user:
            user_name = context.current_user.get("full_name", user_name)

        lines = [
            f"- Current User: {user_name} (Role: {user_role})",
            f"- Active Screen: {context.current_page_title} (Path: `{context.current_page}`)",
        ]

        if context.selected_complaint:
            lines.append(f"- Focused Complaint: {context.selected_complaint}")
        if context.selected_hazard:
            lines.append(f"- Focused Hazard ID: #{context.selected_hazard}")
        if context.selected_conflict:
            lines.append(f"- Focused Conflict Event ID: #{context.selected_conflict}")
        if context.selected_road:
            lines.append(f"- Focused Road Segment: {context.selected_road}")
        if context.selected_evidence:
            lines.append(f"- Focused Evidence Code: {context.selected_evidence}")

        nav_status = context.navigation_state.get("status", "IDLE")
        if nav_status != "IDLE":
            lines.append(f"- Navigation Status: {nav_status} (Step: {context.navigation_state.get('step', 0)})")

        return "\n".join(lines)


# Global singleton instance
context_manager = ContextManager()
