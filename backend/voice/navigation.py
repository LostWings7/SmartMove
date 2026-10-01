"""
SafeCity Loop V2 — FRIDAY Voice Navigation Subsystem
Manages voice-guided safer navigation sessions, step-by-step maneuvers,
hazard alerts along route, and opt-in safety state using RouteEngine.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import time
from sqlalchemy.orm import Session
from backend.services.route_engine import route_engine
from backend.database import SessionLocal


@dataclass
class ManeuverStep:
    step_index: int
    instruction: str
    distance_meters: float
    maneuver_type: str = "straight"
    hazard_warning: Optional[str] = None
    corridor_risk_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "instruction": self.instruction,
            "distance_meters": round(self.distance_meters, 0),
            "maneuver_type": self.maneuver_type,
            "hazard_warning": self.hazard_warning,
            "corridor_risk_score": round(self.corridor_risk_score, 1),
        }


@dataclass
class VoiceNavigationSession:
    session_id: str
    status: str = "IDLE"  # IDLE, ACTIVE, PAUSED, COMPLETED
    origin: str = "School Road [DEMO]"
    destination: str = "Market Junction [DEMO]"
    origin_id: int = 1
    dest_id: int = 2
    route_key: str = "safer"
    route_type: str = "Lower-Risk Route"
    current_step: int = 0
    total_steps: int = 0
    steps: List[ManeuverStep] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    time_minutes: float = 18.0
    distance_km: float = 3.8
    overall_calculated_risk: float = 28.5
    risk_reduction_pct: float = 51.7
    hazards_avoided: int = 4
    high_risk_segments_avoided: int = 3

    def to_dict(self) -> Dict[str, Any]:
        curr_step_obj = self.steps[self.current_step] if self.steps and self.current_step < len(self.steps) else None
        next_step_obj = self.steps[self.current_step + 1] if self.steps and self.current_step + 1 < len(self.steps) else None

        remaining_dist = sum(s.distance_meters for s in self.steps[self.current_step:]) / 1000.0 if self.steps else self.distance_km
        remaining_time = round(remaining_dist * 2.5, 1)

        return {
            "session_id": self.session_id,
            "status": self.status,
            "origin": self.origin,
            "destination": self.destination,
            "origin_id": self.origin_id,
            "dest_id": self.dest_id,
            "route_key": self.route_key,
            "route_type": self.route_type,
            "current_step": self.current_step,
            "total_steps": len(self.steps),
            "time_minutes": self.time_minutes,
            "distance_km": self.distance_km,
            "remaining_time_minutes": remaining_time,
            "remaining_distance_km": round(remaining_dist, 2),
            "overall_calculated_risk": self.overall_calculated_risk,
            "risk_reduction_pct": round(self.risk_reduction_pct, 1),
            "hazards_avoided": self.hazards_avoided,
            "high_risk_segments_avoided": self.high_risk_segments_avoided,
            "current_instruction": curr_step_obj.instruction if curr_step_obj else "Arrived at destination. Safe transit completed.",
            "current_distance_meters": curr_step_obj.distance_meters if curr_step_obj else 0.0,
            "current_maneuver_type": curr_step_obj.maneuver_type if curr_step_obj else "arrive",
            "hazard_warning": curr_step_obj.hazard_warning if curr_step_obj else None,
            "next_instruction": next_step_obj.instruction if next_step_obj else None,
            "next_distance_meters": next_step_obj.distance_meters if next_step_obj else 0.0,
            "steps": [s.to_dict() for s in self.steps],
        }


class VoiceNavigationManager:
    """Manages active turn-by-turn safer navigation sessions powered by RouteEngine."""

    def __init__(self):
        self._sessions: Dict[str, VoiceNavigationSession] = {}

    def get_session(self, session_id: str) -> VoiceNavigationSession:
        if session_id not in self._sessions:
            self._sessions[session_id] = VoiceNavigationSession(session_id=session_id)
        return self._sessions[session_id]

    def start_navigation(
        self,
        session_id: str,
        route_key: str = "safer",
        origin_road_id: int = 1,
        destination_road_id: int = 2,
        db: Optional[Session] = None,
    ) -> VoiceNavigationSession:
        sess = self.get_session(session_id)

        # Query existing RouteEngine for authoritative calculation
        close_db = False
        if db is None:
            db = SessionLocal()
            close_db = True

        try:
            comp = route_engine.compare_routes(origin_road_id, destination_road_id, db)
            fastest_route = next((r for r in comp.routes if r.route_key == "fastest"), comp.routes[0])
            chosen_route = next((r for r in comp.routes if r.route_key == route_key), comp.routes[-1])

            sess.origin = comp.origin_name
            sess.destination = comp.destination_name
            sess.origin_id = origin_road_id
            sess.dest_id = destination_road_id
            sess.route_key = chosen_route.route_key
            sess.route_type = chosen_route.route_type
            sess.time_minutes = chosen_route.time_minutes
            sess.distance_km = chosen_route.distance_km
            sess.overall_calculated_risk = chosen_route.overall_calculated_risk
            sess.status = "ACTIVE"
            sess.current_step = 0
            sess.start_time = time.time()

            # Calculate risk reduction compared to fastest
            if fastest_route.overall_calculated_risk > 0:
                reduction = max(0.0, (fastest_route.overall_calculated_risk - chosen_route.overall_calculated_risk) / fastest_route.overall_calculated_risk * 100.0)
            else:
                reduction = 0.0
            sess.risk_reduction_pct = reduction
            sess.hazards_avoided = max(0, (fastest_route.hazards_count or 4) - (chosen_route.hazards_count or 0))
            sess.high_risk_segments_avoided = max(0, (fastest_route.high_risk_segments_count or 3) - (chosen_route.high_risk_segments_count or 0))

            # Populate maneuvers from RouteEngine
            sess.steps = [
                ManeuverStep(
                    step_index=m.get("step_index", idx),
                    instruction=m.get("instruction", "Continue on route."),
                    distance_meters=float(m.get("distance_meters", 250.0)),
                    maneuver_type=m.get("maneuver_type", "straight"),
                    hazard_warning=m.get("hazard_warning"),
                    corridor_risk_score=float(m.get("corridor_risk", chosen_route.overall_calculated_risk)),
                )
                for idx, m in enumerate(chosen_route.maneuvers)
            ]
            sess.total_steps = len(sess.steps)

            return sess
        finally:
            if close_db:
                db.close()

    def get_navigation_state(self, session_id: str) -> Dict[str, Any]:
        sess = self.get_session(session_id)
        return sess.to_dict()

    def get_next_maneuver(self, session_id: str) -> Dict[str, Any]:
        sess = self.get_session(session_id)
        if not sess.steps or sess.status == "IDLE":
            return {
                "success": False,
                "status": sess.status,
                "instruction": "No active navigation session. Say 'find me a safer route' to begin.",
                "voice_summary": "No active navigation route, Boss. Say 'find me a safer route' to plan a trip.",
            }

        if sess.status == "PAUSED":
            return {
                "success": True,
                "status": "PAUSED",
                "instruction": "Navigation guidance is paused.",
                "voice_summary": "Navigation guidance is paused, Boss. Say 'resume navigation' to continue.",
            }

        if sess.current_step >= len(sess.steps):
            return {
                "success": True,
                "status": "COMPLETED",
                "instruction": f"Arrived at {sess.destination}. Safer navigation complete.",
                "voice_summary": f"You have arrived at {sess.destination}, Boss. Safe transit complete.",
            }

        curr = sess.steps[sess.current_step]
        dist_str = f"{int(curr.distance_meters)} metres" if curr.distance_meters < 1000 else f"{curr.distance_meters/1000.0:.1f} kilometres"
        voice = f"{curr.instruction} ({dist_str})."
        if curr.hazard_warning:
            voice += f" Warning: {curr.hazard_warning}"

        return {
            "success": True,
            "status": sess.status,
            "step_index": sess.current_step,
            "total_steps": len(sess.steps),
            "instruction": curr.instruction,
            "distance_meters": curr.distance_meters,
            "maneuver_type": curr.maneuver_type,
            "hazard_warning": curr.hazard_warning,
            "corridor_risk_score": curr.corridor_risk_score,
            "voice_summary": voice,
        }

    def advance_step(self, session_id: str) -> Dict[str, Any]:
        sess = self.get_session(session_id)
        if sess.current_step < len(sess.steps) - 1:
            sess.current_step += 1
            return self.get_next_maneuver(session_id)
        else:
            sess.status = "COMPLETED"
            return {
                "success": True,
                "status": "COMPLETED",
                "instruction": f"Arrived at {sess.destination}.",
                "voice_summary": f"You have reached your destination at {sess.destination}, Boss.",
            }

    def get_navigation_directions(
        self,
        session_id: str = "default_session",
        start_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Returns the exact displayed navigation maneuvers for sequential voice recitation."""
        sess = self.get_session(session_id)
        if not sess.steps or sess.status == "IDLE":
            return {
                "success": False,
                "status": "IDLE",
                "error": "No active navigation session. Start navigation first.",
                "directions": [],
                "current_index": 0,
                "total": 0,
                "voice_summary": "No active navigation route, Boss. Say 'find me a safer route' to plan a trip.",
            }

        idx = sess.current_step if start_index is None else max(0, min(start_index, len(sess.steps) - 1))
        directions = [
            {
                "index": s.step_index,
                "text": s.instruction,
                "distance_meters": s.distance_meters,
                "maneuver_type": s.maneuver_type,
                "hazard_warning": s.hazard_warning,
            }
            for s in sess.steps
        ]

        dest = sess.destination
        return {
            "success": True,
            "status": sess.status,
            "origin": sess.origin,
            "destination": dest,
            "current_index": sess.current_step,
            "start_index": idx,
            "total": len(directions),
            "directions": directions,
            "voice_summary": f"Reading {len(directions)} navigation directions to {dest}, starting from step {idx + 1}.",
        }

    def pause_navigation(self, session_id: str) -> Dict[str, Any]:
        sess = self.get_session(session_id)
        sess.status = "PAUSED"
        return {
            "success": True,
            "status": "PAUSED",
            "voice_summary": "Navigation paused, Boss.",
            "data": sess.to_dict(),
        }

    def resume_navigation(self, session_id: str) -> Dict[str, Any]:
        sess = self.get_session(session_id)
        sess.status = "ACTIVE"
        curr_inst = sess.steps[sess.current_step].instruction if sess.steps and sess.current_step < len(sess.steps) else "Continue on route."
        return {
            "success": True,
            "status": "ACTIVE",
            "voice_summary": f"Resuming safer navigation. {curr_inst}",
            "data": sess.to_dict(),
        }

    def stop_navigation(self, session_id: str) -> Dict[str, Any]:
        sess = self.get_session(session_id)
        sess.status = "IDLE"
        sess.current_step = 0
        return {
            "success": True,
            "status": "IDLE",
            "voice_summary": "Navigation cancelled, Boss.",
            "data": sess.to_dict(),
        }


navigation_manager = VoiceNavigationManager()
