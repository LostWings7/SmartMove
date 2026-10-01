"""
SafeCity Loop V2 — Voice Assistant (FRIDAY) API Router
Provides token minting for LiveKit WebRTC sessions, context synchronization,
tool enumeration, and direct voice/text query execution.
"""
import os
from typing import Dict, Any, Optional, List
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Body
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field

from backend.database import get_db
from backend.models import User, UserRole
from backend.auth import get_optional_current_user
from backend.voice.context import SafeCityContext, context_manager
from backend.voice.friday_agent import get_friday_agent
from backend.voice.tools import get_tool_registry

router = APIRouter(prefix="/voice", tags=["Voice Assistant (FRIDAY)"])


class VoiceQueryRequest(BaseModel):
    query: str = Field(..., description="Spoken or typed utterance to FRIDAY")
    session_id: str = Field(default="default_session", description="Client session ID")
    context_override: Optional[Dict[str, Any]] = Field(default=None, description="Optional instantaneous context override")


class ContextUpdateRequest(BaseModel):
    session_id: str = Field(default="default_session")
    current_page: Optional[str] = None
    selected_complaint: Optional[str] = None
    selected_hazard: Optional[int] = None
    selected_conflict: Optional[int] = None
    selected_evidence: Optional[str] = None
    selected_road: Optional[str] = None
    map_state: Optional[Dict[str, Any]] = None
    route_state: Optional[Dict[str, Any]] = None
    navigation_state: Optional[Dict[str, Any]] = None


class LiveKitTokenRequest(BaseModel):
    room_name: str = Field(default="safecity-voice-room", description="LiveKit room name")
    identity: Optional[str] = Field(default=None, description="Participant identity")


@router.get("/health")
def voice_health():
    """Returns FRIDAY agent readiness, LiveKit credentials status, and Gemini model config."""
    livekit_url = os.getenv("LIVEKIT_URL", "")
    livekit_key = os.getenv("LIVEKIT_API_KEY", "")
    livekit_secret = os.getenv("LIVEKIT_API_SECRET", "")
    google_key = os.getenv("GOOGLE_API_KEY", "")

    agent = get_friday_agent()
    tools = get_tool_registry()

    return {
        "status": "operational",
        "agent_name": "FRIDAY",
        "livekit_configured": bool(livekit_url and livekit_key and livekit_secret),
        "livekit_url": livekit_url,
        "gemini_configured": bool(google_key),
        "voice_model": os.getenv("FRIDAY_VOICE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025"),
        "voice_name": os.getenv("FRIDAY_VOICE_NAME", "Aoede"),
        "noise_cancellation": "BVC (Background Voice Cancellation)",
        "registered_tools_count": len(tools),
        "is_idle": agent.awaiting_wake,
    }


@router.post("/token")
def generate_livekit_token(
    req: LiveKitTokenRequest = Body(...),
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """
    Mints a cryptographically signed LiveKit AccessToken for WebRTC audio room connection.
    Binds the participant's authenticated user identity and role into token metadata.
    """
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not api_key or not api_secret:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LiveKit credentials (LIVEKIT_API_KEY / LIVEKIT_API_SECRET) are not configured.",
        )

    try:
        from livekit.api import AccessToken, VideoGrants
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="livekit-api library is not installed.",
        )

    identity = req.identity or (f"user_{current_user.id}" if current_user else "citizen_guest")
    name = current_user.full_name if current_user else "Citizen Guest"
    role = current_user.role if current_user else UserRole.CITIZEN

    token = (
        AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(name)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=req.room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_metadata(f'{{"role": "{role}", "user_id": {current_user.id if current_user else "null"}}}')
        .with_ttl(timedelta(hours=2))
    )

    jwt_token = token.to_jwt()

    return {
        "success": True,
        "token": jwt_token,
        "room_name": req.room_name,
        "identity": identity,
        "participant_name": name,
        "role": role,
    }


@router.get("/context")
def get_voice_context(
    session_id: str = "default_session",
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """Retrieves the active UI context sanitized according to caller's role."""
    ctx = context_manager.get_context(session_id)
    role = current_user.role if current_user else ctx.user_role
    sanitized = context_manager.sanitize_for_role(ctx, role)
    return {
        "success": True,
        "session_id": session_id,
        "context": sanitized,
    }


@router.post("/context")
def update_voice_context(
    req: ContextUpdateRequest,
    current_user: Optional[User] = Depends(get_optional_current_user),
):
    """Synchronizes real-time frontend UI state with FRIDAY."""
    updates = req.model_dump(exclude_unset=True)
    session_id = updates.pop("session_id", "default_session")

    if current_user:
        updates["current_user"] = {
            "id": current_user.id,
            "full_name": current_user.full_name,
            "email": current_user.email,
            "role": current_user.role,
        }
        updates["user_role"] = current_user.role
        updates["is_authenticated"] = True

    ctx = context_manager.update_context(session_id, updates)
    return {
        "success": True,
        "session_id": session_id,
        "context": ctx.model_dump(),
    }


@router.post("/query")
def execute_voice_query(
    req: VoiceQueryRequest,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db),
):
    """
    Direct voice/text query interface for testing, automation, and non-WebRTC clients.
    Executes authoritative SafeCity tools on real database data and returns spoken response.
    """
    agent = get_friday_agent()
    ctx = context_manager.get_context(req.session_id)

    # Apply any one-shot context overrides
    if req.context_override:
        context_manager.update_context(req.session_id, req.context_override)
        ctx = context_manager.get_context(req.session_id)

    result = agent.process_query(
        query=req.query,
        context=ctx,
        user=current_user,
        db=db,
    )

    return result


@router.get("/tools")
def list_voice_tools():
    """Lists all registered SafeCity read-only tools and capabilities."""
    tools = get_tool_registry()
    descriptions = {
        "get_current_user": "Returns identity, role, and auth status for current user",
        "get_current_page": "Identifies active screen, route, and focused element in UI",
        "get_dashboard_summary": "Citywide KPIs: monitored roads, active hazards, conflicts, repairs",
        "get_my_complaints": "Retrieves filed citizen complaints for current user",
        "get_complaint_details": "Full complaint record by public reference with RBAC check",
        "get_complaint_status": "Concise lifecycle status and department for a complaint",
        "get_active_hazards": "Active road surface defects (potholes, ponding, cracks)",
        "get_hazard_details": "Detailed hazard telemetry, confidence, and severity by ID",
        "get_traffic_summary": "Surrogate traffic safety metrics and monitored junctions",
        "get_traffic_conflicts": "Surrogate near-miss events with TTC and clearance",
        "get_conflict_details": "Detailed kinematics and collision vector for conflict ID",
        "get_road_conditions": "Surface quality and risk scores for named road segment",
        "get_repair_status": "Municipal repair work orders and engineering status",
        "get_route_summary": "Fastest Route vs. Lower-Risk Safer Route comparison",
        "get_danger_zones": "High-concern corridors and contributing risk factors",
        "calculate_routes": "Calculates Fastest, Balanced, and Safer route alternatives",
        "get_route_details": "Maneuvers, distance, and risk score for a specific route option",
        "compare_routes": "Compares 3 route alternatives using RouteEngine",
        "select_route": "Selects a route alternative for display or pending navigation",
        "start_navigation": "Activates Safe Navigation Mode after explicit user consent",
        "pause_navigation": "Pauses active turn-by-turn guidance",
        "resume_navigation": "Resumes active turn-by-turn guidance",
        "stop_navigation": "Stops or cancels active navigation",
        "get_navigation_state": "Retrieves ETA, remaining distance, maneuvers, and risk state",
        "get_next_maneuver": "Retrieves immediate next driving instruction and distance",
        "get_user_notifications": "Retrieves unread and recent in-app notifications for caller",
        "file_complaint": "Files a new road hazard complaint with category, description, and location",
        "reopen_complaint": "Reopens a resolved or closed complaint for municipal reconsideration",
        "get_unresolved_complaints": "Queries municipal queue for unresolved, pending, or overdue complaints",
        "assign_complaint_to_department": "Assigns municipal complaint to operational department or engineer",
        "update_complaint_status_admin": "Transitions complaint status (reject, verify, resolve, close)",
        "get_cross_domain_intelligence": "Correlates Complaint -> AI Hazard -> Evidence -> Road Risk -> Traffic Conflicts",
        "read_navigation_directions": "Verbatim recitation of displayed turn-by-turn navigation clues",
    }

    tool_list = [
        {"name": name, "description": descriptions.get(name, "SafeCity tactical tool")}
        for name in tools.keys()
    ]

    return {
        "success": True,
        "count": len(tool_list),
        "tools": tool_list,
    }


@router.get("/telemetry")
def get_voice_telemetry(limit: int = 50):
    """Returns real-time execution telemetry and latencies for voice tools."""
    from backend.voice.tools import get_tool_telemetry
    logs = get_tool_telemetry(limit=limit)
    return {
        "success": True,
        "count": len(logs),
        "telemetry": logs,
    }


# ── Navigation REST Endpoints ──────────────────────────────────────────────────

class StartNavigationRequest(BaseModel):
    session_id: str = Field(default="default_session")
    route_key: str = Field(default="safer", description="fastest, balanced, or safer")
    origin_road_id: int = Field(default=1)
    destination_road_id: int = Field(default=2)


class NavigationSessionRequest(BaseModel):
    session_id: str = Field(default="default_session")


@router.get("/navigation/state")
def get_navigation_state(session_id: str = "default_session"):
    """Returns real-time navigation telemetry, current maneuver, and risk reduction."""
    from backend.voice.navigation import navigation_manager
    state = navigation_manager.get_navigation_state(session_id)
    return {"success": True, "navigation_state": state}


@router.post("/navigation/start")
def start_navigation_endpoint(
    req: StartNavigationRequest = Body(...),
    db: Session = Depends(get_db),
):
    """Starts active safe navigation mode along chosen route."""
    from backend.voice.navigation import navigation_manager
    sess = navigation_manager.start_navigation(
        session_id=req.session_id,
        route_key=req.route_key,
        origin_road_id=req.origin_road_id,
        destination_road_id=req.destination_road_id,
        db=db,
    )
    return {
        "success": True,
        "status": sess.status,
        "navigation_state": sess.to_dict(),
        "first_maneuver": sess.steps[0].instruction if sess.steps else "Proceed along route.",
    }


@router.post("/navigation/pause")
def pause_navigation_endpoint(req: NavigationSessionRequest = Body(...)):
    """Pauses active navigation session."""
    from backend.voice.navigation import navigation_manager
    return navigation_manager.pause_navigation(req.session_id)


@router.post("/navigation/resume")
def resume_navigation_endpoint(req: NavigationSessionRequest = Body(...)):
    """Resumes paused navigation session."""
    from backend.voice.navigation import navigation_manager
    return navigation_manager.resume_navigation(req.session_id)


@router.post("/navigation/stop")
def stop_navigation_endpoint(req: NavigationSessionRequest = Body(...)):
    """Cancels active navigation session."""
    from backend.voice.navigation import navigation_manager
    return navigation_manager.stop_navigation(req.session_id)


@router.get("/navigation/next-maneuver")
def get_next_maneuver_endpoint(session_id: str = "default_session"):
    """Returns the upcoming turn-by-turn maneuver instruction and distance."""
    from backend.voice.navigation import navigation_manager
    return navigation_manager.get_next_maneuver(session_id)


@router.post("/navigation/advance")
def advance_navigation_step_endpoint(req: NavigationSessionRequest = Body(...)):
    """Advances one maneuver step (GPS progression / demo simulation)."""
    from backend.voice.navigation import navigation_manager
    return navigation_manager.advance_step(req.session_id)


@router.get("/navigation/directions")
def get_navigation_directions_endpoint(
    session_id: str = "default_session",
    start_index: Optional[int] = None,
):
    """Returns exact displayed navigation clues for recitation."""
    from backend.voice.navigation import navigation_manager
    return navigation_manager.get_navigation_directions(session_id=session_id, start_index=start_index)
