// SafeCity Loop V2 — Voice Action Dispatcher & Allowlisted RPC Engine
// Validates, routes, and safely executes FRIDAY's controlled frontend operations.

import type { NavigateFunction } from 'react-router-dom'
import type { VoiceAction, VoiceActionType } from '@/types/voice'

export const PAGE_ALLOWLIST: Record<string, string> = {
  'dashboard': '/',
  'overview': '/',
  '/': '/',
  'my-complaints': '/citizen/complaints',
  'complaints': '/citizen/complaints',
  '/citizen/complaints': '/citizen/complaints',
  'new-complaint': '/citizen/complaints/new',
  '/citizen/complaints/new': '/citizen/complaints/new',
  'complaint-details': '/citizen/complaints',
  'ai-vision': '/ai-vision',
  '/ai-vision': '/ai-vision',
  'road-intelligence': '/road-intelligence',
  '/road-intelligence': '/road-intelligence',
  'traffic-conflicts': '/traffic-conflicts',
  '/traffic-conflicts': '/traffic-conflicts',
  'hazards': '/map',
  'map': '/map',
  '/map': '/map',
  'repairs': '/repair',
  'repair': '/repair',
  '/repair': '/repair',
  'reports': '/reports',
  '/reports': '/reports',
  'danger-zones': '/danger-zones',
  '/danger-zones': '/danger-zones',
  'routes': '/routes',
  '/routes': '/routes',
  'evidence': '/evidence',
  '/evidence': '/evidence',
  'junction': '/junction',
  '/junction': '/junction',
  'analytics': '/analytics',
  '/analytics': '/analytics',
  'settings': '/settings',
  '/settings': '/settings',
  'admin-dashboard': '/admin/dashboard',
  '/admin/dashboard': '/admin/dashboard',
  'admin-complaints': '/admin/complaints',
  '/admin/complaints': '/admin/complaints',
  'admin-users': '/admin/users',
  '/admin/users': '/admin/users',
  'notifications': '/citizen/notifications',
  '/citizen/notifications': '/citizen/notifications',
}

const ADMIN_ONLY_PAGES = new Set([
  '/admin/dashboard',
  '/admin/complaints',
  '/admin/users',
])

const STAFF_ROLES = new Set(['INSPECTOR', 'ENGINEER', 'ADMIN', 'SUPERUSER'])

type ActionHandler = (payload: Record<string, any>) => boolean | Promise<boolean>

class VoiceActionDispatcher {
  private listeners: Map<VoiceActionType, Set<ActionHandler>> = new Map()

  /**
   * Components (like SafeCityMap, TrafficConflicts, AIVision) subscribe to specific actions.
   * Returns an unsubscribe cleanup function.
   */
  public subscribe(action: VoiceActionType, handler: ActionHandler): () => void {
    if (!this.listeners.has(action)) {
      this.listeners.set(action, new Set())
    }
    const handlers = this.listeners.get(action)!
    handlers.add(handler)

    return () => {
      handlers.delete(handler)
      if (handlers.size === 0) {
        this.listeners.delete(action)
      }
    }
  }

  /**
   * Resolves a human or semantic page name to an authorized SafeCity route.
   */
  public resolvePagePath(pageName: string): string | null {
    const clean = (pageName || '').trim().toLowerCase()
    if (PAGE_ALLOWLIST[clean]) {
      return PAGE_ALLOWLIST[clean]
    }
    if (clean.startsWith('/citizen/complaints/') || clean.startsWith('/admin/complaints/')) {
      return clean
    }
    return null
  }

  /**
   * Dispatches an allowlisted voice action through registered handlers and navigation.
   */
  public async dispatch(
    action: VoiceAction,
    navigate: NavigateFunction,
    userRole: string = 'CITIZEN'
  ): Promise<{ success: boolean; error?: string }> {
    const { action: actionName, payload = {} } = action
    const roleUpper = (userRole || 'CITIZEN').toUpperCase()

    // 1. Notify any active component listeners first (e.g. Map zoom or Video seek)
    let handledByListener = false
    const handlers = this.listeners.get(actionName)
    if (handlers && handlers.size > 0) {
      for (const handler of Array.from(handlers)) {
        try {
          const res = await handler(payload)
          if (res) handledByListener = true
        } catch (err) {
          console.error(`[VoiceDispatcher] Handler error for '${actionName}':`, err)
        }
      }
    }

    // 2. Perform global routing and action handling
    switch (actionName) {
      case 'navigate_to_page': {
        const rawTarget = payload.path || payload.page || ''
        const resolvedPath = this.resolvePagePath(rawTarget)
        if (!resolvedPath) {
          return { success: false, error: `Page '${rawTarget}' is not an authorized SafeCity route.` }
        }
        if (ADMIN_ONLY_PAGES.has(resolvedPath) && !STAFF_ROLES.has(roleUpper)) {
          return { success: false, error: `Access Denied: Citizens cannot access admin route '${resolvedPath}'.` }
        }
        navigate(resolvedPath)
        return { success: true }
      }

      case 'open_complaint': {
        const ref = payload.reference || ''
        const compId = payload.id
        const targetPath = payload.path || (
          STAFF_ROLES.has(roleUpper)
            ? (compId ? `/admin/complaints/${compId}` : '/admin/complaints')
            : (compId ? `/citizen/complaints/${compId}` : '/citizen/complaints')
        )
        navigate(targetPath)
        return { success: true }
      }

      case 'open_hazard': {
        const hId = payload.id || payload.hazard_id
        navigate(`/map?hazard=${hId}`)
        return { success: true }
      }

      case 'open_conflict': {
        const confId = payload.id || payload.conflict_id
        if (!handledByListener) {
          navigate(`/traffic-conflicts?event=${confId}`)
        }
        return { success: true }
      }

      case 'open_evidence': {
        const evId = payload.id || ''
        navigate(`/evidence?code=${encodeURIComponent(evId)}`)
        return { success: true }
      }

      case 'focus_map': {
        if (!handledByListener) {
          const lat = payload.lat
          const lng = payload.lng
          const zoom = payload.zoom || 16
          navigate(`/map?lat=${lat}&lng=${lng}&zoom=${zoom}`)
        }
        return { success: true }
      }

      case 'highlight_hazard':
      case 'highlight_map_entity': {
        if (!handledByListener) {
          const id = payload.id
          navigate(`/map?highlight=${id}`)
        }
        return { success: true }
      }

      case 'highlight_complaint': {
        const ref = payload.reference || ''
        if (!handledByListener) {
          navigate(`/map?complaint=${encodeURIComponent(ref)}`)
        }
        return { success: true }
      }

      case 'seek_video':
      case 'seek_video_to_event': {
        const secs = payload.timestamp_seconds ?? payload.seconds ?? 0
        if (!handledByListener) {
          // Navigate to traffic conflicts with seek param
          navigate(`/traffic-conflicts?seek=${secs}`)
        }
        return { success: true }
      }

      case 'open_route':
      case 'show_route':
      case 'show_route_comparison': {
        navigate('/routes')
        return { success: true }
      }

      case 'show_danger_zones': {
        navigate('/danger-zones')
        return { success: true }
      }

      case 'show_status_history': {
        const ref = payload.reference
        if (payload.id) {
          navigate(`/citizen/complaints/${payload.id}#history`)
        } else {
          navigate(`/citizen/complaints?reference=${encodeURIComponent(ref)}`)
        }
        return { success: true }
      }

      case 'select_route': {
        if (!handledByListener) {
          const rKey = payload.route_key || 'safer'
          navigate(`/routes?route=${rKey}`)
        }
        return { success: true }
      }

      case 'start_navigation': {
        if (!handledByListener) {
          const rKey = payload.route_key || 'safer'
          navigate(`/routes?nav=active&route=${rKey}`)
        }
        return { success: true }
      }

      case 'pause_navigation':
      case 'resume_navigation':
      case 'stop_navigation': {
        if (!handledByListener) {
          navigate('/routes')
        }
        return { success: true }
      }

      case 'read_navigation_directions':
      case 'read_next_navigation_direction':
      case 'stop_navigation_speech': {
        if (!handledByListener) {
          navigate('/routes')
        }
        return { success: true }
      }

      case 'open_notifications':
      case 'show_notifications': {
        navigate('/citizen/notifications')
        return { success: true }
      }

      case 'open_admin_complaints': {
        if (!STAFF_ROLES.has(roleUpper)) {
          return { success: false, error: 'Access Denied: Citizens cannot access admin complaints queue.' }
        }
        navigate('/admin/complaints')
        return { success: true }
      }

      case 'confirm_action':
      case 'cancel_action': {
        return { success: true }
      }

      default:
        if (handledByListener) {
          return { success: true }
        }
        return { success: false, error: `Action '${actionName}' could not be dispatched.` }
    }
  }
}

export const voiceActionDispatcher = new VoiceActionDispatcher()
