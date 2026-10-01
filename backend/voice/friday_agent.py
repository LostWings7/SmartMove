"""
SafeCity Loop V2 — FRIDAY Voice Agent Core Subsystem
Integrates LiveKit Agents, Google Gemini Realtime audio, noise cancellation,
SafeCity tool dispatch, and emotional tone adaptation.
"""
import os
import re
import asyncio
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import google, noise_cancellation

from backend.database import SessionLocal
from backend.models import User
from backend.voice.context import SafeCityContext, context_manager
from backend.voice.permissions import VoicePermissions
from backend.voice.prompt import (
    SAFECITY_FRIDAY_SYSTEM_PROMPT,
    detect_emotion,
    EMOTION_STYLES,
    build_friday_prompt,
)
from backend.voice.tools import safecity_voice_tools, get_tool_registry

load_dotenv()

# Realtime Model Configuration
DEFAULT_VOICE_MODEL = os.getenv("FRIDAY_VOICE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
DEFAULT_VOICE_NAME = os.getenv("FRIDAY_VOICE_NAME", "Aoede")


class FridayAgent(Agent):
    """
    FRIDAY Real-Time AI Tactical Voice Assistant for SafeCity Loop V2.
    Supports LiveKit WebRTC sessions and standalone Python/API query execution.
    """

    def __init__(self, instructions: str = SAFECITY_FRIDAY_SYSTEM_PROMPT, session_id: str = "default_session", **kwargs):
        super().__init__(instructions=instructions)
        self.session_id: str = session_id
        self.awaiting_wake: bool = False
        self.last_active: float = 0.0
        self.memory: Dict[str, Any] = {}
        try:
            loop = asyncio.get_event_loop()
            self.last_active = loop.time()
        except RuntimeError:
            import time
            self.last_active = time.time()

    async def on_session_start(self, session: AgentSession):
        """Called when a LiveKit room session begins."""
        asyncio.create_task(self.monitor_idle(session))

    async def monitor_idle(self, session: AgentSession):
        """Monitors conversational inactivity and enters idle state after 60s."""
        while True:
            await asyncio.sleep(5)
            if self.awaiting_wake:
                continue
            try:
                now = asyncio.get_event_loop().time()
            except RuntimeError:
                import time
                now = time.time()

            if now - self.last_active > 60:
                self.awaiting_wake = True
                print("💤 FRIDAY has gone idle. Say 'Friday' to wake her.")

    async def on_message(self, ctx: AgentSession, message: str) -> str:
        """Handles spoken utterances from LiveKit audio stream."""
        try:
            now = asyncio.get_event_loop().time()
        except RuntimeError:
            import time
            now = time.time()

        msg_clean = message.strip().lower()

        # Handle explicit sleep command
        if "sleep now" in msg_clean or "go to sleep" in msg_clean:
            self.awaiting_wake = True
            print("😴 FRIDAY is going to sleep on command.")
            await ctx.speak("Going idle, Boss.")
            return ""

        # Handle wake word
        if self.awaiting_wake:
            if "friday" in msg_clean:
                self.awaiting_wake = False
                self.last_active = now
                await ctx.speak("I'm here, Boss. How can I assist with SafeCity?")
                return ""
            return ""

        if ctx and hasattr(ctx, "is_speaking") and ctx.is_speaking:
            await ctx.stop_speaking()

        self.last_active = now

        # Detect emotion and style prompt
        emotion = detect_emotion(message)
        tone_instruction = EMOTION_STYLES.get(emotion, EMOTION_STYLES["neutral"])

        # Augment with SafeCity live context
        active_context = context_manager.get_context()
        styled_prompt = build_friday_prompt(
            user_query=message,
            context=active_context,
            tool_results=None,
        )

        return styled_prompt

    def process_query(
        self,
        query: str,
        context: Optional[SafeCityContext] = None,
        user: Optional[User] = None,
        db=None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Authoritative direct query execution for testing, HTTP REST API, and RPC.
        Resolves query intent to SafeCity tools, queries real database, and formats voice response.
        """
        sid = session_id or (context.client_id if context and getattr(context, "client_id", None) else getattr(self, "session_id", "default_session"))
        ctx = context or context_manager.get_context(sid)
        if user:
            ctx.current_user = {
                "id": user.id,
                "full_name": user.full_name,
                "email": user.email,
                "role": user.role,
            }
            ctx.user_role = user.role
            ctx.is_authenticated = True

        q_lower = query.strip().lower()
        emotion = detect_emotion(query)

        # Sleep / Wake state handling
        if "sleep now" in q_lower or "go to sleep" in q_lower:
            self.awaiting_wake = True
            return {
                "success": True,
                "reply": "Going idle, Boss. Say 'Friday' whenever you need me.",
                "emotion": "tactical",
                "is_idle": True,
                "tools_called": [],
                "frontend_actions": [],
            }

        if self.awaiting_wake:
            if "friday" in q_lower:
                self.awaiting_wake = False
                return {
                    "success": True,
                    "reply": "I'm right here, Boss. Ready for SafeCity operations.",
                    "emotion": "friendly",
                    "is_idle": False,
                    "tools_called": [],
                    "frontend_actions": [],
                }
            return {
                "success": True,
                "reply": "FRIDAY is currently idle. Say 'Friday' to wake her.",
                "emotion": "neutral",
                "is_idle": True,
                "tools_called": [],
                "frontend_actions": [],
            }

        tools_registry = get_tool_registry()
        tools_called: List[Dict[str, Any]] = []
        frontend_actions: List[Dict[str, Any]] = []
        spoken_reply = ""
        raw_tool_data = None

        # ── 0A. Pending Consequential Action Confirmation / Cancellation (Phase 4.1 & 4.4) ──
        if ctx.pending_action:
            words = set(re.findall(r"\b\w+\b", q_lower))
            is_negative = (
                bool(words.intersection({"no", "n", "cancel", "stop", "abort", "dont", "never"}))
                or "don't" in q_lower
                or "never mind" in q_lower
                or "do not" in q_lower
            )
            is_affirmative = (
                not is_negative
                and (
                    bool(words.intersection({"yes", "y", "sure", "ok", "okay", "proceed", "confirm", "submit", "affirmative"}))
                    or any(p in q_lower for p in ["submit it", "confirm it", "go ahead", "do it", "do that", "let's do it", "yes please"])
                )
            )

            if is_affirmative:
                act = ctx.pending_action
                act_name = act.get("action")
                act_args = act.get("args", {})
                ctx.pending_action = None
                ctx.pending_complaint_draft = None

                if act_name == "file_complaint":
                    res = tools_registry["file_complaint"](
                        category=act_args.get("category", "pothole"),
                        description=act_args.get("description", "Hazard observed"),
                        location_address=act_args.get("location_address", "Market Junction"),
                        session_id=sid,
                        context=ctx,
                        db=db,
                    )
                    tools_called.append({"tool": "file_complaint", "args": act_args, "result": res})
                    raw_tool_data = res
                    spoken_reply = res.get("voice_summary", "Complaint submitted successfully, Boss.")
                    if res.get("success"):
                        ctx.selected_complaint = res.get("reference")
                        frontend_actions.append({
                            "action": "open_complaint",
                            "payload": {"reference": res.get("reference"), "id": res.get("id", 1), "path": f"/citizen/complaints/{res.get('id', 1)}"}
                        })
                elif act_name == "reopen_complaint":
                    res = tools_registry["reopen_complaint"](
                        reference=act_args.get("reference", ctx.selected_complaint or "SC-2026-000184"),
                        reason=act_args.get("reason", "Citizen requested reopening"),
                        session_id=sid,
                        context=ctx,
                        db=db,
                    )
                    tools_called.append({"tool": "reopen_complaint", "args": act_args, "result": res})
                    raw_tool_data = res
                    spoken_reply = res.get("voice_summary", "Complaint reopened successfully, Boss.")
                elif act_name == "update_complaint_status_admin":
                    res = tools_registry["update_complaint_status_admin"](
                        reference=act_args.get("reference", ctx.selected_complaint or "SC-2026-000184"),
                        new_status=act_args.get("new_status", "REJECTED"),
                        comment=act_args.get("comment", "Updated via FRIDAY voice assistant"),
                        session_id=sid,
                        context=ctx,
                        db=db,
                    )
                    tools_called.append({"tool": "update_complaint_status_admin", "args": act_args, "result": res})
                    raw_tool_data = res
                    spoken_reply = res.get("voice_summary", "Status updated successfully, Boss.")
                elif act_name == "assign_complaint_to_department":
                    res = tools_registry["assign_complaint_to_department"](
                        reference=act_args.get("reference", ctx.selected_complaint or "SC-2026-000184"),
                        department=act_args.get("department", "Road Engineering"),
                        session_id=sid,
                        context=ctx,
                        db=db,
                    )
                    tools_called.append({"tool": "assign_complaint_to_department", "args": act_args, "result": res})
                    raw_tool_data = res
                    spoken_reply = res.get("voice_summary", "Complaint assigned successfully, Boss.")
                else:
                    spoken_reply = "Action confirmed and executed, Boss."

            elif is_negative:
                ctx.pending_action = None
                ctx.pending_complaint_draft = None
                spoken_reply = "Action cancelled, Boss. No changes were made."
                frontend_actions.append({"action": "cancel_action", "payload": {}})

        # ── 0B. Conversational Complaint Gathering (Draft Location Step) ──
        elif ctx.pending_complaint_draft:
            draft = ctx.pending_complaint_draft
            cat = draft.get("category", "pothole")
            loc = None
            if any(p in q_lower for p in ["current location", "my location", "here", "map location", "yes use current location"]):
                loc = ctx.map_state.get("center_name") or "Market Junction"
            else:
                for r_name in ["college junction", "school road", "market junction", "hospital road", "industrial road", "main road"]:
                    if r_name in q_lower:
                        loc = r_name.title()
                        break
                if not loc:
                    loc = query.strip(".,!? ").title()

            ctx.pending_complaint_draft = None
            ctx.pending_action = {
                "action": "file_complaint",
                "args": {
                    "category": cat,
                    "location_address": loc,
                    "description": f"Voice reported {cat} hazard near {loc}."
                }
            }
            spoken_reply = f"Your {cat} complaint near {loc} is ready. Shall I submit it, Boss?"
            frontend_actions.append({
                "action": "confirm_action",
                "payload": {"action": "file_complaint", "category": cat, "location": loc}
            })

        # ── 0C. Conversational Complaint Filing Intent (Phase 4.1) ──
        elif any(p in q_lower for p in ["file a complaint", "report a pothole", "report waterlogging", "report a hazard", "report road damage", "report ponding", "report defect", "create a complaint"]):
            if "waterlogging" in q_lower or "ponding" in q_lower:
                category = "waterlogging"
            elif "crack" in q_lower:
                category = "cracks"
            elif "traffic" in q_lower:
                category = "traffic_hazard"
            elif "pothole" in q_lower:
                category = "pothole"
            else:
                category = "pothole"

            loc = None
            for r_name in ["college junction", "school road", "market junction", "hospital road", "industrial road", "main road"]:
                if r_name in q_lower:
                    loc = r_name.title()
                    break

            if loc:
                ctx.pending_action = {
                    "action": "file_complaint",
                    "args": {
                        "category": category,
                        "location_address": loc,
                        "description": f"Voice reported {category} hazard near {loc}."
                    }
                }
                spoken_reply = f"Your {category} complaint near {loc} is ready. Shall I submit it, Boss?"
                frontend_actions.append({
                    "action": "confirm_action",
                    "payload": {"action": "file_complaint", "category": category, "location": loc}
                })
            else:
                ctx.pending_complaint_draft = {"category": category}
                spoken_reply = f"Sure, Boss. Where is the {category} located?"

        # ── 0D. Reopen Complaint Intent (Phase 4.1) ──
        elif any(p in q_lower for p in ["reopen this complaint", "reopen complaint", "re-open complaint"]):
            match = re.search(r"sc[-\w]+", q_lower, re.IGNORECASE)
            ref = match.group(0).upper() if match else (ctx.selected_complaint or "SC-2026-000184")
            ctx.pending_action = {
                "action": "reopen_complaint",
                "args": {"reference": ref, "reason": "Citizen requested reopening via voice assistant"}
            }
            spoken_reply = f"This will reopen complaint {ref}. Are you sure you want to proceed, Boss?"
            frontend_actions.append({
                "action": "confirm_action",
                "payload": {"action": "reopen_complaint", "reference": ref}
            })

        # ── 0E. Notifications Intent (Phase 4.7) ──
        elif any(p in q_lower for p in ["my notifications", "read my notifications", "show my notifications", "any notifications", "check notifications", "read notifications"]):
            res = tools_registry["get_user_notifications"](limit=5, unread_only=False, session_id=sid, context=ctx, db=db)
            tools_called.append({"tool": "get_user_notifications", "args": {"limit": 5}, "result": res})
            spoken_reply = res["voice_summary"]
            raw_tool_data = res
            frontend_actions.append({"action": "navigate_to_page", "payload": {"path": "/citizen/notifications"}})
            frontend_actions.append({"action": "open_notifications", "payload": {"notifications": res.get("notifications", [])}})

        # ── 0F. Repair Status & Repair Evidence Intent ──
        elif any(p in q_lower for p in ["has the repair been completed", "is the repair completed", "repair completed"]):
            ref = ctx.selected_complaint or ctx.selected_road or "School Road"
            res = tools_registry["get_repair_status"](reference=ref, db=db)
            tools_called.append({"tool": "get_repair_status", "args": {"reference": ref}, "result": res})
            spoken_reply = res["voice_summary"]
            raw_tool_data = res

        elif any(p in q_lower for p in ["show the repair evidence", "repair evidence"]):
            ref = ctx.selected_complaint or "SC-2026-000184"
            ev_code = f"EV-REPAIR-{ref}"
            frontend_actions.append({"action": "open_evidence", "payload": {"id": ev_code, "reference": ref, "type": "repair"}})
            spoken_reply = f"Displaying verified post-repair completion evidence {ev_code} for complaint {ref}, Boss."

        # ── 0G. Admin Reject Complaint (Phase 4.4 - Consequential Action Confirmation) ──
        elif any(p in q_lower for p in ["reject this complaint", "reject complaint"]):
            if ctx.user_role == "CITIZEN":
                spoken_reply = "Access denied, Boss. Complaint rejection and status mutations are restricted to authorized municipal staff."
            else:
                match = re.search(r"sc[-\w]+", q_lower, re.IGNORECASE)
                ref = match.group(0).upper() if match else (ctx.selected_complaint or "SC-2026-000184")
                ctx.pending_action = {
                    "action": "update_complaint_status_admin",
                    "args": {"reference": ref, "new_status": "REJECTED", "comment": "Rejected via FRIDAY voice action"}
                }
                spoken_reply = f"This will mark complaint {ref} as REJECTED. Do you want me to continue, Boss?"
                frontend_actions.append({
                    "action": "confirm_action",
                    "payload": {"action": "update_complaint_status_admin", "reference": ref, "new_status": "REJECTED"}
                })

        # ── 0H. Admin Assign Complaint to Department (Phase 4.4) ──
        elif any(p in q_lower for p in ["assign this complaint to", "assign complaint to", "assign to road engineering", "assign to drainage"]):
            if ctx.user_role == "CITIZEN":
                spoken_reply = "Access denied, Boss. Complaint assignments are restricted to authorized municipal staff."
            else:
                match = re.search(r"sc[-\w]+", q_lower, re.IGNORECASE)
                ref = match.group(0).upper() if match else (ctx.selected_complaint or "SC-2026-000184")
                dept = "Road Engineering"
                if "drainage" in q_lower or "waterlogging" in q_lower:
                    dept = "Drainage & Flood Management"
                elif "traffic" in q_lower:
                    dept = "Traffic Operations"
                elif "electrical" in q_lower or "lighting" in q_lower:
                    dept = "Electrical & Lighting"

                ctx.pending_action = {
                    "action": "assign_complaint_to_department",
                    "args": {"reference": ref, "department": dept}
                }
                spoken_reply = f"This will assign complaint {ref} to {dept}. Shall I proceed, Boss?"
                frontend_actions.append({
                    "action": "confirm_action",
                    "payload": {"action": "assign_complaint_to_department", "reference": ref, "department": dept}
                })

        # ── 0I. Admin Unresolved, Overdue & Department Queues (Phase 4.3) ──
        elif any(p in q_lower for p in ["unresolved complaints", "pending complaints", "overdue complaints", "complaints assigned to", "pending pothole complaints", "unresolved waterlogging", "show today's unresolved complaints", "how many pothole complaints are pending"]):
            if ctx.user_role == "CITIZEN":
                spoken_reply = "Access denied, Boss. Unresolved complaint queues and internal operational records are restricted to municipal staff and administrators."
            else:
                dept = None
                if "road engineering" in q_lower or "pothole" in q_lower:
                    dept = "Road Engineering"
                elif "drainage" in q_lower or "waterlogging" in q_lower:
                    dept = "Drainage & Flood Management"
                elif "traffic" in q_lower:
                    dept = "Traffic Operations"

                overdue = "overdue" in q_lower
                res = tools_registry["get_unresolved_complaints"](department=dept, overdue_only=overdue, limit=10, session_id=sid, context=ctx, db=db)
                tools_called.append({"tool": "get_unresolved_complaints", "args": {"department": dept, "overdue_only": overdue}, "result": res})
                spoken_reply = res["voice_summary"]
                raw_tool_data = res
                frontend_actions.append({"action": "open_admin_complaints", "payload": {"department": dept, "overdue_only": overdue}})

        # ── 0J. Admin Open Latest Critical Complaint (Phase 4.3) ──
        elif any(p in q_lower for p in ["open latest critical complaint", "latest critical complaint"]):
            if ctx.user_role == "CITIZEN":
                spoken_reply = "Access denied, Boss. Municipal critical complaint queues are restricted to staff."
            else:
                res = tools_registry["get_unresolved_complaints"](priority="CRITICAL", limit=1, session_id=sid, context=ctx, db=db)
                tools_called.append({"tool": "get_unresolved_complaints", "args": {"priority": "CRITICAL", "limit": 1}, "result": res})
                raw_tool_data = res
                if res.get("complaints"):
                    crit = res["complaints"][0]
                    ctx.selected_complaint = crit["reference"]
                    frontend_actions.append({
                        "action": "open_complaint",
                        "payload": {"reference": crit["reference"], "id": crit["id"], "path": f"/admin/complaints/{crit['id']}"}
                    })
                    spoken_reply = f"Opening latest critical complaint {crit['reference']} regarding '{crit['title']}' at {crit['location']}, Boss."
                else:
                    spoken_reply = "No critical unresolved complaints currently on file, Boss."

        # ── 0K. Cross-Domain Intelligence (Phase 4.6) ──
        elif any(p in q_lower for p in ["what evidence do we have", "looks serious", "cross domain", "cross-domain", "is this road currently dangerous", "is this road dangerous"]):
            ref = ctx.selected_complaint or "SC-2026-000184"
            road = ctx.selected_road or "School Road"
            res = tools_registry["get_cross_domain_intelligence"](complaint_reference=ref, road_name=road, session_id=sid, db=db)
            tools_called.append({"tool": "get_cross_domain_intelligence", "args": {"complaint_reference": ref, "road_name": road}, "result": res})
            spoken_reply = res["voice_summary"]
            raw_tool_data = res
            frontend_actions.append({"action": "navigate_to_page", "payload": {"path": "/road-intelligence"}})

        # ── 1. Page / Screen Intent ──
        elif any(p in q_lower for p in ["what page", "where am i", "current page", "which page", "current screen", "active screen"]):
            res = tools_registry["get_current_page"](context=ctx)
            tools_called.append({"tool": "get_current_page", "args": {}, "result": res})
            spoken_reply = res["voice_summary"]
            raw_tool_data = res

        # ── 2. User Identity Intent ──
        elif any(p in q_lower for p in ["who am i", "current user", "my profile", "my role", "logged in as"]):
            res = tools_registry["get_current_user"](context=ctx)
            tools_called.append({"tool": "get_current_user", "args": {}, "result": res})
            spoken_reply = res["voice_summary"]
            raw_tool_data = res

        # ── 3. Complaint Details & Open Complaint Intent ──
        elif any(p in q_lower for p in ["open complaint", "open that complaint", "view complaint"]):
            match = re.search(r"sc[-\w]+", q_lower, re.IGNORECASE)
            ref = match.group(0).upper() if match else ctx.selected_complaint or "SC-2026-000184"
            res = tools_registry["get_complaint_details"](complaint_reference=ref, context=ctx, db=db)
            tools_called.append({"tool": "get_complaint_details", "args": {"complaint_reference": ref}, "result": res})
            raw_tool_data = res
            if res.get("success"):
                ctx.selected_complaint = res["reference"]
                target_path = f"/citizen/complaints/{res.get('id', 1)}" if ctx.user_role == "CITIZEN" else f"/admin/complaints/{res.get('id', 1)}"
                frontend_actions.append({
                    "action": "open_complaint",
                    "payload": {"reference": res["reference"], "id": res.get("id", 1), "path": target_path}
                })
                spoken_reply = f"Opening complaint {res['reference']} regarding '{res['title']}' for you now, Boss."
            else:
                spoken_reply = res["voice_summary"]

        # ── 4. Complaint Status History Intent ──
        elif any(p in q_lower for p in ["status history", "history of complaint", "timeline of complaint", "show me the status history"]):
            ref = ctx.selected_complaint or "SC-2026-000184"
            res = tools_registry["get_complaint_details"](complaint_reference=ref, context=ctx, db=db)
            tools_called.append({"tool": "get_complaint_details", "args": {"complaint_reference": ref}, "result": res})
            raw_tool_data = res
            frontend_actions.append({"action": "show_status_history", "payload": {"reference": ref}})
            spoken_reply = f"Displaying the complete lifecycle status history for complaint {ref}, Boss."

        # ── 5. Evidence for Complaint Intent ──
        elif any(p in q_lower for p in ["evidence for that complaint", "complaint evidence", "evidence for complaint"]):
            ref = ctx.selected_complaint or "SC-2026-000184"
            res = tools_registry["get_complaint_details"](complaint_reference=ref, context=ctx, db=db)
            tools_called.append({"tool": "get_complaint_details", "args": {"complaint_reference": ref}, "result": res})
            raw_tool_data = res
            ev_code = "EV-2026-0001"
            frontend_actions.append({"action": "open_evidence", "payload": {"id": ev_code, "reference": ref}})
            spoken_reply = f"Retrieving official verified evidence {ev_code} linked to complaint {ref}, Boss."

        # ── 6. Complaint Status Intent ──
        elif any(p in q_lower for p in ["status of complaint", "complaint status", "status of sc-", "how is complaint", "what's happening with my complaint", "what is happening with my complaint", "status of my complaint", "status of this complaint"]):
            match = re.search(r"sc[-\w]+", q_lower, re.IGNORECASE)
            ref = match.group(0).upper() if match else ctx.selected_complaint or "SC-2026-000184"
            res = tools_registry["get_complaint_status"](complaint_reference=ref, context=ctx, db=db)
            tools_called.append({"tool": "get_complaint_status", "args": {"complaint_reference": ref}, "result": res})
            spoken_reply = res["voice_summary"]
            raw_tool_data = res
            if res.get("success"):
                frontend_actions.append({"action": "open_complaint", "payload": {"reference": res["reference"]}})

        # ── 7. My Complaints Intent ──
        elif any(p in q_lower for p in ["my complaints", "show my complaints", "open my complaints", "view complaints", "filed complaints", "list complaints"]):
            res = tools_registry["get_my_complaints"](context=ctx, db=db)
            tools_called.append({"tool": "get_my_complaints", "args": {}, "result": res})
            spoken_reply = res["voice_summary"]
            raw_tool_data = res
            frontend_actions.append({"action": "navigate_to_page", "payload": {"path": "/citizen/complaints"}})

        # ── 8. Critical Traffic Conflict Intent ──
        elif any(p in q_lower for p in ["critical conflict", "show me the critical conflict", "most severe conflict"]):
            res = tools_registry["get_traffic_conflicts"](limit=1, min_severity="CRITICAL", db=db)
            tools_called.append({"tool": "get_traffic_conflicts", "args": {"limit": 1, "min_severity": "CRITICAL"}, "result": res})
            raw_tool_data = res
            if res.get("conflicts"):
                top_conf = res["conflicts"][0]
                ctx.selected_conflict = top_conf["id"]
                frontend_actions.append({"action": "open_conflict", "payload": {"id": top_conf["id"], "ttc": top_conf.get("ttc")}})
                spoken_reply = (
                    f"Displaying critical traffic conflict #{top_conf['id']} at {top_conf['location']} "
                    f"involving {top_conf['road_users']} with a TTC of {top_conf['ttc']}s, Boss."
                )
            else:
                spoken_reply = "Boss, no critical traffic conflict events are currently flagged."

        # ── 9. Specific Conflict Lookup / Open Conflict Intent ──
        elif any(p in q_lower for p in ["open conflict", "show conflict", "conflict nm-"]):
            # Extract ID e.g. NM-1216 or 1216 or 2
            match = re.search(r"(\d+)", q_lower)
            conf_id = int(match.group(1)) if match else (ctx.selected_conflict or 1)
            res = tools_registry["get_conflict_details"](conflict_id=conf_id, db=db)
            tools_called.append({"tool": "get_conflict_details", "args": {"conflict_id": conf_id}, "result": res})
            raw_tool_data = res
            if res.get("success"):
                ctx.selected_conflict = conf_id
                frontend_actions.append({"action": "open_conflict", "payload": {"id": conf_id}})
                spoken_reply = f"Opening traffic conflict event #{conf_id} at {res['location']}, Boss."
            else:
                # Fallback to general conflicts list
                c_list = tools_registry["get_traffic_conflicts"](limit=1, db=db)
                if c_list.get("conflicts"):
                    c_id = c_list["conflicts"][0]["id"]
                    ctx.selected_conflict = c_id
                    frontend_actions.append({"action": "open_conflict", "payload": {"id": c_id}})
                    spoken_reply = f"Opening traffic conflict event #{c_id} for you now, Boss."
                else:
                    spoken_reply = f"Traffic conflict event #{conf_id} could not be found, Boss."

        # ── 10. Conflict Evidence Intent ──
        elif any(p in q_lower for p in ["show me its evidence", "show conflict evidence", "open conflict evidence", "its evidence"]):
            conf_id = ctx.selected_conflict or 2
            res = tools_registry["get_conflict_details"](conflict_id=conf_id, db=db)
            tools_called.append({"tool": "get_conflict_details", "args": {"conflict_id": conf_id}, "result": res})
            raw_tool_data = res
            ev_code = res.get("evidence_code") or f"EV-CONFLICT-{conf_id}"
            frontend_actions.append({
                "action": "open_evidence",
                "payload": {"id": ev_code, "conflict_id": conf_id, "clip_url": res.get("evidence_clip")}
            })
            spoken_reply = f"Displaying verified surrogate conflict video evidence {ev_code} for event #{conf_id}, Boss."

        # ── 11. Video Seeking / Moment in Video Intent ──
        elif any(p in q_lower for p in ["take me to that moment", "seek video", "jump to video", "moment in the video"]):
            conf_id = ctx.selected_conflict or 2
            res = tools_registry["get_conflict_details"](conflict_id=conf_id, db=db)
            tools_called.append({"tool": "get_conflict_details", "args": {"conflict_id": conf_id}, "result": res})
            raw_tool_data = res
            target_sec = res.get("timestamp_seconds", 14.5)
            clip_url = res.get("evidence_clip", "/samples/sample_traffic.mp4")
            frontend_actions.append({
                "action": "seek_video",
                "payload": {
                    "timestamp_seconds": target_sec,
                    "seconds": target_sec,
                    "conflict_id": conf_id,
                    "clip_url": clip_url,
                }
            })
            spoken_reply = f"Seeking video playback to {target_sec:.1f} seconds at the exact moment of minimum time-to-collision for conflict #{conf_id}, Boss."

        # ── 12. Traffic Conflicts Navigation Intent ──
        elif any(p in q_lower for p in ["open traffic conflicts", "traffic conflict", "active conflict", "near miss", "near-miss", "ttc"]):
            res = tools_registry["get_traffic_conflicts"](limit=5, db=db)
            tools_called.append({"tool": "get_traffic_conflicts", "args": {"limit": 5}, "result": res})
            spoken_reply = res["voice_summary"]
            raw_tool_data = res
            frontend_actions.append({"action": "navigate_to_page", "payload": {"path": "/traffic-conflicts"}})

        # ── 13. AI Vision Potholes Navigation Intent ──
        elif any(p in q_lower for p in ["pothole detections", "show me the pothole detections", "pothole inspection"]):
            res = tools_registry["get_active_hazards"](limit=5, hazard_type="pothole", db=db)
            tools_called.append({"tool": "get_active_hazards", "args": {"hazard_type": "pothole"}, "result": res})
            spoken_reply = f"Switching to AI Vision pothole inspection viewport, Boss. Found {res['total_found']} active potholes."
            raw_tool_data = res
            frontend_actions.append({"action": "navigate_to_page", "payload": {"path": "/ai-vision", "tab": "potholes"}})

        # ── 14. AI Vision Waterlogging / Ponding Intent ──
        elif any(p in q_lower for p in ["waterlogging analysis", "open the waterlogging analysis", "ponding analysis"]):
            res = tools_registry["get_active_hazards"](limit=5, hazard_type="waterlogging", db=db)
            tools_called.append({"tool": "get_active_hazards", "args": {"hazard_type": "waterlogging"}, "result": res})
            spoken_reply = "Opening SafeCity waterlogging and ponding segmentation analytics, Boss."
            raw_tool_data = res
            frontend_actions.append({"action": "navigate_to_page", "payload": {"path": "/road-intelligence", "tab": "waterlogging"}})

        # ── 15. Hazard Evidence Intent ──
        elif any(p in q_lower for p in ["evidence for this hazard", "hazard evidence", "show hazard evidence"]):
            h_id = ctx.selected_hazard or 1
            res = tools_registry["get_hazard_details"](hazard_id=h_id, db=db)
            tools_called.append({"tool": "get_hazard_details", "args": {"hazard_id": h_id}, "result": res})
            raw_tool_data = res
            ev_code = res.get("evidence_code") or f"EV-HAZARD-{h_id}"
            frontend_actions.append({"action": "open_evidence", "payload": {"id": ev_code, "hazard_id": h_id}})
            spoken_reply = f"Opening verified inspection evidence {ev_code} for hazard #{h_id}, Boss."

        # ── 16. Map Focus Intent ──
        elif any(p in q_lower for p in ["focus map", "focus the map", "center map"]):
            # Extract road name e.g. School Road or Market Junction
            road_name = "School Road"
            if "market" in q_lower:
                road_name = "Market Junction"
            elif "hospital" in q_lower:
                road_name = "Hospital Road"
            elif "industrial" in q_lower:
                road_name = "Industrial Road"

            cond_res = tools_registry["get_road_conditions"](location=road_name, db=db)
            tools_called.append({"tool": "get_road_conditions", "args": {"location": road_name}, "result": cond_res})
            raw_tool_data = cond_res

            # Nominal coordinates
            coords = {"School Road": (51.5074, -0.1278), "Market Junction": (51.5120, -0.1190), "Hospital Road": (51.5030, -0.1350), "Industrial Road": (51.5200, -0.1100)}
            lat, lng = coords.get(road_name, (51.5074, -0.1278))
            frontend_actions.append({
                "action": "focus_map",
                "payload": {"lat": lat, "lng": lng, "zoom": 16, "location": road_name}
            })
            spoken_reply = f"Focusing the SafeCity GIS map on {road_name} at coordinates ({lat:.4f}, {lng:.4f}), Boss."

        # ── 17. Highlight Hazard / Complaint Intent ──
        elif any(p in q_lower for p in ["highlight hazard", "highlight the hazard"]):
            match = re.search(r"(\d+)", q_lower)
            h_id = int(match.group(1)) if match else (ctx.selected_hazard or 1)
            frontend_actions.append({"action": "highlight_hazard", "payload": {"id": h_id}})
            spoken_reply = f"Highlighting hazard #{h_id} on the active map, Boss."

        elif any(p in q_lower for p in ["highlight complaint", "highlight the complaint"]):
            match = re.search(r"sc[-\w]+", q_lower, re.IGNORECASE)
            ref = match.group(0).upper() if match else (ctx.selected_complaint or "SC-2026-000184")
            frontend_actions.append({"action": "highlight_complaint", "payload": {"reference": ref}})
            spoken_reply = f"Highlighting complaint {ref} location on the active map, Boss."

        # ── 18. Dangerous Areas / Danger Zones Intent ──
        elif any(p in q_lower for p in ["dangerous area", "danger zone", "high risk corridor", "hotspot", "unsafe road", "dangerous road", "show danger zones"]):
            res = tools_registry["get_danger_zones"](limit=5, db=db)
            tools_called.append({"tool": "get_danger_zones", "args": {"limit": 5}, "result": res})
            spoken_reply = res["voice_summary"]
            raw_tool_data = res
            frontend_actions.append({"action": "show_danger_zones", "payload": {}})

        # ── 18A. Explicit Opt-in / Start Safer Navigation Intent ──
        # CRUCIAL RULE: Never automatically start navigation without explicit user opt-in.
        elif (
            any(p == q_lower.strip(".,!? ") for p in ["yes", "yes please", "sure", "ok", "okay", "yes use the safer route", "yes start navigation", "affirmative", "let's go"])
            and getattr(ctx, "pending_route_opt_in", False)
        ) or any(p in q_lower for p in ["start navigation", "start safer navigation", "use the safer route", "use safer route", "begin navigation", "enable navigation"]):
            clean_route = getattr(ctx, "selected_route", "safer") or "safer"
            res = tools_registry["start_navigation"](route_id=clean_route, session_id=sid, context=ctx, db=db)
            tools_called.append({"tool": "start_navigation", "args": {"route_id": clean_route}, "result": res})
            ctx.pending_route_opt_in = False
            raw_tool_data = res
            spoken_reply = res["voice_summary"]
            frontend_actions.append({
                "action": "start_navigation",
                "payload": {"route_key": clean_route, "navigation_state": res.get("navigation_state", {})}
            })

        # ── 18B. Pause Navigation Intent ──
        elif any(p in q_lower for p in ["pause navigation", "pause route", "pause guidance"]) or (q_lower.strip(".,!? ") == "pause"):
            res = tools_registry["pause_navigation"](session_id=sid)
            tools_called.append({"tool": "pause_navigation", "args": {}, "result": res})
            spoken_reply = res["voice_summary"]
            raw_tool_data = res
            frontend_actions.append({"action": "pause_navigation", "payload": {}})

        # ── 18C. Resume Navigation Intent ──
        elif any(p in q_lower for p in ["resume navigation", "resume route", "resume guidance", "continue navigation"]) or (q_lower.strip(".,!? ") == "resume"):
            res = tools_registry["resume_navigation"](session_id=sid)
            tools_called.append({"tool": "resume_navigation", "args": {}, "result": res})
            spoken_reply = res["voice_summary"]
            raw_tool_data = res
            frontend_actions.append({"action": "resume_navigation", "payload": {}})

        # ── 18D. Stop / Cancel Navigation Intent ──
        elif any(p in q_lower for p in ["stop navigation", "cancel navigation", "end navigation", "exit navigation", "quit navigation"]):
            res = tools_registry["stop_navigation"](session_id=sid)
            tools_called.append({"tool": "stop_navigation", "args": {}, "result": res})
            spoken_reply = res["voice_summary"]
            raw_tool_data = res
            frontend_actions.append({"action": "stop_navigation", "payload": {}})

        # ── 18D1. Stop Navigation Recitation Intent ──
        elif any(p in q_lower for p in ["stop reading", "stop reciting", "stop reading directions", "stop reading navigation", "stop speaking directions", "stop voice directions", "stop direction speech", "stop reading maneuvers"]):
            frontend_actions.append({"action": "stop_navigation_speech", "payload": {}})
            spoken_reply = "Stopped reading directions, Boss."

        # ── 18D2. Read All / Remaining Navigation Directions Intent ──
        elif any(p in q_lower for p in [
            "read navigation directions",
            "read the navigation directions",
            "read directions",
            "read the directions",
            "read all directions",
            "read all the directions",
            "read route directions",
            "recite directions",
            "recite the directions",
            "recite navigation",
            "read maneuvers",
            "read the maneuvers",
            "read out the directions",
            "read out directions",
            "read turn by turn",
        ]) or (("read" in q_lower or "recite" in q_lower) and ("direction" in q_lower or "maneuver" in q_lower) and "next" not in q_lower and "stop" not in q_lower):
            res = tools_registry["read_navigation_directions"](session_id=sid)
            tools_called.append({"tool": "read_navigation_directions", "args": {}, "result": res})
            raw_tool_data = res
            spoken_reply = res["voice_summary"]
            if res.get("success"):
                frontend_actions.append({
                    "action": "read_navigation_directions",
                    "payload": {
                        "start_index": res.get("start_index", 0),
                        "total": res.get("total", 0),
                        "directions": res.get("directions", []),
                    }
                })

        # ── 18D3. Read Next Navigation Direction Intent ──
        elif any(p in q_lower for p in ["read next turn", "read next direction", "read next maneuver", "recite next turn", "read next clue", "recite next direction"]):
            res = tools_registry["get_next_maneuver"](session_id=sid)
            tools_called.append({"tool": "get_next_maneuver", "args": {}, "result": res})
            raw_tool_data = res
            instruction = res.get("instruction", "Continue on route.")
            spoken_reply = instruction
            frontend_actions.append({
                "action": "read_next_navigation_direction",
                "payload": {"instruction": instruction}
            })

        # ── 18E. Next Turn / Maneuver Guidance Intent (Driver Safety - concise) ──
        elif any(p in q_lower for p in ["next turn", "what's the next turn", "what is the next turn", "which turn", "next maneuver", "next instruction", "how far until the next turn", "distance to next turn", "how far to the next turn"]):
            res = tools_registry["get_next_maneuver"](session_id=sid)
            tools_called.append({"tool": "get_next_maneuver", "args": {}, "result": res})
            raw_tool_data = res
            if any(p in q_lower for p in ["how far", "distance"]):
                dist = res.get("distance_meters", 0)
                dist_str = f"{int(dist)} metres" if dist < 1000 else f"{dist/1000.0:.1f} kilometres"
                spoken_reply = f"{dist_str} until: {res.get('instruction', 'Proceed along route')}."
            else:
                spoken_reply = res.get("voice_summary", res.get("instruction", "Continue on route."))

        # ── 18F. Navigation ETA / Remaining Time Intent ──
        elif any(p in q_lower for p in ["how much longer", "how long until", "remaining time", "what is the eta", "what's the eta", "time remaining"]):
            res = tools_registry["get_navigation_state"](session_id=sid)
            tools_called.append({"tool": "get_navigation_state", "args": {}, "result": res})
            raw_tool_data = res
            nav_state = res.get("navigation_state", {})
            rem_time = nav_state.get("remaining_time_minutes", 18.0)
            spoken_reply = f"Approximately {rem_time:.0f} minutes remaining to {nav_state.get('destination', 'destination')}, Boss."

        # ── 18G. Risk Avoided Query Intent ──
        elif any(p in q_lower for p in ["how much risk does this route avoid", "how much risk is reduced", "how much risk", "what risk does this avoid", "risk avoided", "risk reduction"]):
            res = tools_registry["get_navigation_state"](session_id=sid)
            tools_called.append({"tool": "get_navigation_state", "args": {}, "result": res})
            raw_tool_data = res
            nav_state = res.get("navigation_state", {})
            pct = nav_state.get("risk_reduction_pct", 51.7)
            segs = nav_state.get("high_risk_segments_avoided", 3)
            spoken_reply = f"This route reduces calculated risk by {pct:.1f}% and avoids {segs} high-risk road segments, Boss."

        # ── 18H. Select Route Option Intent ──
        elif any(p in q_lower for p in ["select fastest route", "choose fastest route", "fastest route option", "show me the fastest route", "switch to fastest"]):
            res = tools_registry["select_route"](route_id="fastest", session_id=sid, context=ctx, db=db)
            tools_called.append({"tool": "select_route", "args": {"route_id": "fastest"}, "result": res})
            ctx.selected_route = "fastest"
            spoken_reply = "Selected the fastest arterial route, Boss. Estimated duration is 14 minutes with higher risk exposure."
            raw_tool_data = res
            frontend_actions.append({"action": "select_route", "payload": {"route_key": "fastest"}})

        elif any(p in q_lower for p in ["select balanced route", "choose balanced route", "balanced route option", "show me the balanced route", "switch to balanced"]):
            res = tools_registry["select_route"](route_id="balanced", session_id=sid, context=ctx, db=db)
            tools_called.append({"tool": "select_route", "args": {"route_id": "balanced"}, "result": res})
            ctx.selected_route = "balanced"
            spoken_reply = "Selected the balanced route option, Boss. Moderated speed and reduced hazard exposure."
            raw_tool_data = res
            frontend_actions.append({"action": "select_route", "payload": {"route_key": "balanced"}})

        elif any(p in q_lower for p in ["select safer route", "choose safer route", "safer route option", "show me the safer route", "switch to safer"]):
            res = tools_registry["select_route"](route_id="safer", session_id=sid, context=ctx, db=db)
            tools_called.append({"tool": "select_route", "args": {"route_id": "safer"}, "result": res})
            ctx.selected_route = "safer"
            spoken_reply = "Selected the safer bypass route, Boss. Optimized for minimum risk exposure."
            raw_tool_data = res
            frontend_actions.append({"action": "select_route", "payload": {"route_key": "safer"}})

        # ── 19. Route Calculation / Find Safer Route Intent (Phase 3.3) ──
        elif any(p in q_lower for p in ["find me a safer route", "find a safer route", "safer route", "fastest route", "calculate route", "calculate routes", "compare route", "compare routes", "route comparison", "show route comparison", "show route", "navigation", "route summary"]):
            res = tools_registry["calculate_routes"](origin=None, destination=None, db=db)
            tools_called.append({"tool": "calculate_routes", "args": {}, "result": res})
            ctx.pending_route_opt_in = True
            ctx.selected_route = "safer"
            spoken_reply = res["voice_summary"]
            raw_tool_data = res
            frontend_actions.append({"action": "show_route_comparison", "payload": {"routes": res.get("routes", [])}})
            frontend_actions.append({"action": "select_route", "payload": {"route_key": "safer"}})

        # ── 20. Road Condition / Surface Hazards Intent ──
        elif any(p in q_lower for p in ["road condition", "pothole", "cracks", "waterlogging", "surface damage", "road quality"]):
            loc = None
            for r_name in ["school road", "market junction", "hospital road", "industrial road", "main road"]:
                if r_name in q_lower:
                    loc = r_name
                    break
            if loc:
                res = tools_registry["get_road_conditions"](location=loc, db=db)
                tools_called.append({"tool": "get_road_conditions", "args": {"location": loc}, "result": res})
            else:
                res = tools_registry["get_active_hazards"](limit=5, db=db)
                tools_called.append({"tool": "get_active_hazards", "args": {"limit": 5}, "result": res})
            spoken_reply = res["voice_summary"]
            raw_tool_data = res
            frontend_actions.append({"action": "navigate_to_page", "payload": {"path": "/road-intelligence"}})

        # ── 21. Dashboard / City Overview Intent ──
        elif any(p in q_lower for p in ["dashboard", "overview", "city summary", "citywide status", "city stats"]):
            res = tools_registry["get_dashboard_summary"](db=db)
            tools_called.append({"tool": "get_dashboard_summary", "args": {}, "result": res})
            spoken_reply = res["voice_summary"]
            raw_tool_data = res

        # ── 22. Repair Work Order Intent ──
        elif any(p in q_lower for p in ["repair", "work order", "resurfacing", "maintenance"]):
            res = tools_registry["get_repair_status"](reference="School Road", db=db)
            tools_called.append({"tool": "get_repair_status", "args": {"reference": "School Road"}, "result": res})
            spoken_reply = res["voice_summary"]
            raw_tool_data = res
            frontend_actions.append({"action": "navigate_to_page", "payload": {"path": "/repair"}})

        # ── Default Tactical Response ──
        else:
            res = tools_registry["get_dashboard_summary"](db=db)
            tools_called.append({"tool": "get_dashboard_summary", "args": {}, "result": res})
            spoken_reply = (
                f"Boss, standing by for SafeCity operations. "
                f"Currently tracking {res['roads_monitored']} road segments and {res['active_hazards']} active hazards. "
                f"Ask me about your complaints, road conditions, traffic conflicts, or safer routes."
            )
            raw_tool_data = res

        # Filter frontend actions through VoicePermissions
        valid_actions = []
        for act in frontend_actions:
            is_valid, err = VoicePermissions.validate_action(
                action_name=act["action"],
                payload=act.get("payload", {}),
                user_role=ctx.user_role,
                user_id=ctx.current_user.get("id") if ctx.current_user else None,
            )
            if is_valid:
                valid_actions.append(act)

        return {
            "success": True,
            "reply": spoken_reply,
            "emotion": emotion,
            "is_idle": False,
            "tools_called": tools_called,
            "frontend_actions": valid_actions,
            "data": raw_tool_data,
        }


# Global singleton instance
_agent_instance: Optional[FridayAgent] = None


def get_friday_agent() -> FridayAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = FridayAgent()
    return _agent_instance


# ── LiveKit Worker Entrypoint ─────────────────────────────────────────────────

async def entrypoint(ctx: agents.JobContext):
    """LiveKit Agents Worker Entrypoint."""
    assistant = get_friday_agent()

    realtime_model = os.getenv("FRIDAY_VOICE_MODEL", DEFAULT_VOICE_MODEL)
    voice_name = os.getenv("FRIDAY_VOICE_NAME", DEFAULT_VOICE_NAME)

    session = AgentSession(
        llm=google.beta.realtime.RealtimeModel(
            model=realtime_model,
            voice=voice_name,
            temperature=0.7,
            instructions=SAFECITY_FRIDAY_SYSTEM_PROMPT,
        ),
        tools=safecity_voice_tools,
    )

    await session.start(
        room=ctx.room,
        agent=assistant,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
