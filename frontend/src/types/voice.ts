// SafeCity Loop V2 — FRIDAY Voice Assistant Types & Allowlisted Actions

export type VoiceActionType =
  | 'navigate_to_page'
  | 'open_complaint'
  | 'open_hazard'
  | 'open_conflict'
  | 'open_evidence'
  | 'focus_map'
  | 'highlight_map_entity'
  | 'highlight_hazard'
  | 'highlight_complaint'
  | 'seek_video'
  | 'seek_video_to_event'
  | 'open_route'
  | 'show_route'
  | 'show_route_comparison'
  | 'show_dashboard_card'
  | 'show_status_history'
  | 'show_danger_zones'
  | 'select_route'
  | 'start_navigation'
  | 'pause_navigation'
  | 'resume_navigation'
  | 'stop_navigation'
  | 'read_navigation_directions'
  | 'read_next_navigation_direction'
  | 'stop_navigation_speech'
  | 'open_notifications'
  | 'show_notifications'
  | 'open_admin_complaints'
  | 'confirm_action'
  | 'cancel_action'

export interface VoiceAction {
  action: VoiceActionType
  payload: Record<string, any>
}

export interface VoiceToolExecution {
  tool: string
  args: Record<string, any>
  result: Record<string, any>
}

export interface VoiceQueryResult {
  success: boolean
  reply: string
  emotion: 'urgent' | 'friendly' | 'supportive' | 'tactical' | 'neutral'
  is_idle: boolean
  tools_called: VoiceToolExecution[]
  frontend_actions: VoiceAction[]
  data?: any
  error?: string
}

export interface SafeCityContextState {
  client_id?: string
  current_user?: {
    id: number
    full_name: string
    email: string
    role: string
  } | null
  user_role?: string
  is_authenticated?: boolean
  current_page: string
  current_page_title?: string
  selected_complaint?: string | null
  selected_hazard?: number | null
  selected_conflict?: number | null
  selected_evidence?: string | null
  selected_road?: string | null
  map_state?: Record<string, any>
  route_state?: Record<string, any>
  navigation_state?: {
    status: 'IDLE' | 'ACTIVE' | 'PAUSED' | 'COMPLETED'
    step?: number
  }
}
