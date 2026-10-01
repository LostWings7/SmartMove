"""
SafeCity Loop V2 — Voice Assistant Subsystem (FRIDAY)
Real-time voice intelligence, LiveKit WebRTC agent, role-aware context,
and safe allowlisted SafeCity tool RPC layer.
"""
from backend.voice.context import SafeCityContext, context_manager
from backend.voice.permissions import VoicePermissions
from backend.voice.prompt import SAFECITY_FRIDAY_SYSTEM_PROMPT, detect_emotion
from backend.voice.tools import safecity_voice_tools, get_tool_registry
from backend.voice.friday_agent import FridayAgent, get_friday_agent

__all__ = [
    "SafeCityContext",
    "context_manager",
    "VoicePermissions",
    "SAFECITY_FRIDAY_SYSTEM_PROMPT",
    "detect_emotion",
    "safecity_voice_tools",
    "get_tool_registry",
    "FridayAgent",
    "get_friday_agent",
]
