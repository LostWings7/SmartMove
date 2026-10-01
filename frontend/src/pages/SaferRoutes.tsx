import { useState, useEffect, useCallback, useRef } from 'react'
import { useSearchParams } from 'react-router-dom'
import Layout from '@/components/layout/Layout'
import DemoDataBanner from '@/components/ui/DemoDataBanner'
import { useRoads, useRouteComparison } from '@/hooks/useApi'
import type { RouteOption } from '@/types'
import { voiceActionDispatcher } from '@/services/voiceActionDispatcher'
import { useVoice } from '@/context/VoiceContext'

function riskColor(score: number): string {
  if (score >= 70) return '#ef4444'
  if (score >= 45) return '#f59e0b'
  return '#10b981'
}

function riskBadge(classification: string) {
  const cls =
    classification === 'VERY HIGH'
      ? 'bg-red-900/50 text-red-300 border-red-700'
      : classification === 'HIGH'
      ? 'bg-orange-900/50 text-orange-300 border-orange-700'
      : classification === 'MEDIUM'
      ? 'bg-amber-900/50 text-amber-300 border-amber-700'
      : 'bg-emerald-900/50 text-emerald-300 border-emerald-700'
  return <span className={`text-xs font-bold px-2 py-0.5 rounded border ${cls}`}>{classification}</span>
}

function getManeuverIcon(type?: string): string {
  switch (type) {
    case 'turn-right':
    case 'right':
      return '↱'
    case 'turn-left':
    case 'left':
      return '↰'
    case 'fork':
    case 'merge':
      return '⤳'
    case 'arrive':
      return '🏁'
    default:
      return '↑'
  }
}

interface NavigationState {
  session_id: string
  status: 'IDLE' | 'ACTIVE' | 'PAUSED' | 'COMPLETED'
  origin: string
  destination: string
  origin_id: number
  dest_id: number
  route_key: string
  route_type: string
  current_step: number
  total_steps: number
  time_minutes: number
  distance_km: number
  remaining_time_minutes: number
  remaining_distance_km: number
  overall_calculated_risk: number
  risk_reduction_pct: number
  hazards_avoided: number
  high_risk_segments_avoided: number
  current_instruction: string
  current_distance_meters: number
  current_maneuver_type: string
  hazard_warning?: string | null
  next_instruction?: string | null
  next_distance_meters?: number
  steps?: Array<{
    step_index: number
    instruction: string
    distance_meters: number
    maneuver_type: string
    hazard_warning?: string | null
  }>
}

function SafeNavigationHUD({
  nav,
  onPause,
  onResume,
  onStop,
  onAdvance,
  recitation,
  onReadDirections,
  onReadNext,
  onStopRecitation,
}: {
  nav: NavigationState
  onPause: () => void
  onResume: () => void
  onStop: () => void
  onAdvance: () => void
  recitation: { isReading: boolean; currentSpeechIndex: number; total: number }
  onReadDirections: (startIndex?: number) => void
  onReadNext: () => void
  onStopRecitation: () => void
}) {
  const isPaused = nav.status === 'PAUSED'

  return (
    <div className="mb-6 rounded-2xl border-2 border-emerald-500/80 bg-gradient-to-r from-emerald-950/80 via-navy-900 to-navy-950 p-5 shadow-2xl shadow-emerald-950/50">
      {/* Header telemetry row */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-emerald-800/40 pb-4">
        <div className="flex items-center gap-3">
          <span className="relative flex h-3.5 w-3.5">
            <span
              className={`absolute inline-flex h-full w-full rounded-full opacity-75 ${
                isPaused ? 'bg-amber-400' : 'animate-ping bg-emerald-400'
              }`}
            />
            <span
              className={`relative inline-flex h-3.5 w-3.5 rounded-full ${
                isPaused ? 'bg-amber-500' : 'bg-emerald-500'
              }`}
            />
          </span>

          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-sm font-black tracking-wider text-emerald-300">
                SAFE ROUTE ACTIVE
              </span>
              <span
                className={`rounded border px-2 py-0.5 font-mono text-[11px] font-bold ${
                  isPaused
                    ? 'border-amber-600 bg-amber-900/60 text-amber-200'
                    : 'border-emerald-600 bg-emerald-900/60 text-emerald-200'
                }`}
              >
                {nav.status}
              </span>
              <span className="rounded bg-navy-800 px-2 py-0.5 font-mono text-[10px] text-gray-300 border border-gray-700">
                FRIDAY VOICE GUIDED
              </span>
            </div>
            <div className="mt-0.5 text-xs text-gray-300">
              En Route to <strong className="text-white">{nav.destination}</strong> via {nav.route_type}
            </div>
          </div>
        </div>

        {/* Telemetry quick stats */}
        <div className="flex flex-wrap items-center gap-3 text-right font-mono">
          <div className="rounded-lg border border-emerald-800/50 bg-emerald-950/60 px-3 py-1.5">
            <div className="text-[10px] uppercase text-emerald-400">Remaining ETA</div>
            <div className="text-lg font-bold text-white">
              {nav.remaining_time_minutes.toFixed(0)}{' '}
              <span className="text-xs font-normal text-emerald-300">min</span>
            </div>
          </div>

          <div className="rounded-lg border border-gray-800 bg-navy-900 px-3 py-1.5">
            <div className="text-[10px] uppercase text-gray-400">Distance Left</div>
            <div className="text-lg font-bold text-white">
              {nav.remaining_distance_km.toFixed(1)}{' '}
              <span className="text-xs font-normal text-gray-400">km</span>
            </div>
          </div>

          <div className="rounded-lg border border-emerald-800/50 bg-emerald-950/60 px-3 py-1.5">
            <div className="text-[10px] uppercase text-emerald-400">Risk Avoidance</div>
            <div className="text-lg font-bold text-emerald-300">
              🛡 -{nav.risk_reduction_pct.toFixed(0)}%
            </div>
          </div>

          <div className="rounded-lg border border-gray-800 bg-navy-900 px-3 py-1.5">
            <div className="text-[10px] uppercase text-gray-400">Hazards Bypassed</div>
            <div className="text-lg font-bold text-amber-300">
              {nav.hazards_avoided} Defects
            </div>
          </div>
        </div>
      </div>

      {/* Maneuver & Driving Instruction Row */}
      <div className="mt-4 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-start gap-4">
          <div className="flex h-14 w-14 shrink-0 items-center justify-center rounded-xl bg-emerald-500/20 text-3xl font-black text-emerald-300 border border-emerald-500/40">
            {getManeuverIcon(nav.current_maneuver_type)}
          </div>

          <div>
            <div className="text-[11px] font-mono uppercase tracking-wider text-emerald-400">
              Step {nav.current_step + 1} of {nav.total_steps || 3} — In{' '}
              {Math.round(nav.current_distance_meters)} meters
            </div>
            <h2 className="text-lg md:text-xl font-bold text-white mt-0.5">
              {nav.current_instruction}
            </h2>

            {nav.hazard_warning && (
              <div className="mt-1.5 inline-flex items-center gap-1.5 rounded-md border border-amber-600/60 bg-amber-950/40 px-2.5 py-0.5 text-xs text-amber-300">
                <span>⚠️ Tactical Advisory:</span>
                <span>{nav.hazard_warning}</span>
              </div>
            )}

            {nav.next_instruction && (
              <div className="mt-1 text-xs text-gray-400">
                Then: <span className="text-gray-300 font-medium">{nav.next_instruction}</span>{' '}
                ({Math.round(nav.next_distance_meters || 0)} m)
              </div>
            )}
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2 shrink-0 w-full md:w-auto justify-end">
          <button
            onClick={onAdvance}
            className="rounded-lg border border-gray-700 bg-navy-800 px-3 py-2 text-xs font-semibold text-gray-200 hover:bg-navy-700 transition"
            title="Advance one maneuver step (GPS simulation)"
          >
            ⏭ Next Turn
          </button>

          {isPaused ? (
            <button
              onClick={onResume}
              className="rounded-lg border border-emerald-600 bg-emerald-600 px-4 py-2 text-xs font-bold text-white hover:bg-emerald-500 transition shadow"
            >
              ▶ Resume Guidance
            </button>
          ) : (
            <button
              onClick={onPause}
              className="rounded-lg border border-amber-600/70 bg-amber-900/40 px-4 py-2 text-xs font-bold text-amber-200 hover:bg-amber-900/60 transition"
            >
              ⏸ Pause
            </button>
          )}

          <button
            onClick={onStop}
            className="rounded-lg border border-red-800/80 bg-red-950/50 px-3 py-2 text-xs font-semibold text-red-300 hover:bg-red-900/70 transition"
          >
            ⏹ Cancel Navigation
          </button>
        </div>
      </div>

      {/* FRIDAY Voice Recitation Controls */}
      <div className="mt-3.5 pt-3 border-t border-emerald-800/40 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono uppercase tracking-wider text-emerald-400 flex items-center gap-1.5">
            <span
              className={`inline-block h-2 w-2 rounded-full ${
                recitation.isReading ? 'bg-emerald-400 animate-pulse' : 'bg-gray-500'
              }`}
            />
            FRIDAY Voice Recitation:
          </span>
          {recitation.isReading ? (
            <span className="text-xs font-mono text-emerald-300 font-semibold bg-emerald-950/70 border border-emerald-700/60 rounded px-2.5 py-0.5 animate-pulse">
              🔊 Reading Step {recitation.currentSpeechIndex + 1} of {recitation.total}
            </span>
          ) : (
            <span className="text-xs text-gray-400">Verbatim turn-by-turn guidance</span>
          )}
        </div>

        <div className="flex items-center gap-2">
          {recitation.isReading ? (
            <button
              onClick={onStopRecitation}
              className="rounded-lg border border-red-600/70 bg-red-900/60 px-3.5 py-1.5 text-xs font-bold text-red-200 hover:bg-red-800 transition flex items-center gap-1.5 shadow"
            >
              <span>■ Stop Reading</span>
            </button>
          ) : (
            <button
              onClick={() => onReadDirections(nav.current_step)}
              className="rounded-lg border border-emerald-500/80 bg-emerald-700/60 px-3.5 py-1.5 text-xs font-bold text-emerald-100 hover:bg-emerald-600/80 transition flex items-center gap-1.5 shadow"
            >
              <span>🔊 Read Directions</span>
            </button>
          )}

          <button
            onClick={onReadNext}
            className="rounded-lg border border-gray-700 bg-navy-800 px-3.5 py-1.5 text-xs font-semibold text-gray-200 hover:bg-navy-700 transition flex items-center gap-1.5"
            title="Recite the immediate driving instruction"
          >
            <span>▶ Read Next</span>
          </button>
        </div>
      </div>
    </div>
  )
}

function RouteCard({
  route,
  isSelected,
  onSelect,
  onStartNavigation,
}: {
  route: RouteOption
  isSelected: boolean
  onSelect: () => void
  onStartNavigation: () => void
}) {
  const rc = riskColor(route.overall_calculated_risk)
  const isFastest = route.route_key === 'fastest'
  const isBalanced = route.route_key === 'balanced'
  const isSafer = route.route_key === 'safer'

  return (
    <div
      onClick={onSelect}
      className={`card relative cursor-pointer p-5 border-2 transition-all ${
        isSelected
          ? 'border-emerald-500 bg-navy-800/95 ring-2 ring-emerald-500/40 shadow-xl shadow-emerald-950/30'
          : isSafer
          ? 'border-emerald-600/40 bg-emerald-950/10 hover:border-emerald-500/60'
          : isBalanced
          ? 'border-amber-700/40 bg-amber-950/10 hover:border-amber-600/60'
          : 'border-gray-800 bg-navy-800 hover:border-gray-700'
      }`}
    >
      {/* Selected Route Pill */}
      {isSelected && (
        <div className="absolute -top-3 right-4 rounded-full bg-emerald-500 px-3 py-0.5 font-mono text-[10px] font-extrabold uppercase text-navy-950 shadow">
          ✓ Active Route Selection
        </div>
      )}

      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-white font-bold text-base">{route.route_type.toUpperCase()}</h3>
            {isFastest && (
              <span className="bg-sky-900/60 text-sky-300 border border-sky-700 text-[10px] px-2 py-0.5 rounded font-mono">
                ⏱ Minimum Duration
              </span>
            )}
            {isBalanced && (
              <span className="bg-amber-900/60 text-amber-300 border border-amber-700 text-[10px] px-2 py-0.5 rounded font-mono">
                ⚖ Speed & Safety Balance
              </span>
            )}
            {isSafer && (
              <span className="bg-emerald-900/60 text-emerald-300 border border-emerald-700 text-[10px] px-2 py-0.5 rounded font-mono">
                🛡 Safety Optimized
              </span>
            )}
          </div>
          <p className="text-gray-400 text-xs mt-1">{route.description}</p>
        </div>

        <div className="text-right">
          <span
            className="text-xs font-bold px-2.5 py-1 rounded inline-block font-mono"
            style={{ backgroundColor: rc + '22', color: rc, border: `1px solid ${rc}55` }}
          >
            Risk Exposure: {Math.round(route.overall_calculated_risk)}/100
          </span>
          <div className="mt-1">{riskBadge(route.risk_classification)}</div>
        </div>
      </div>

      {/* 4 Core Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        <div className="bg-navy-900 p-3 rounded-lg border border-gray-800/80">
          <div className="text-[11px] text-gray-500 uppercase font-mono">Travel Time</div>
          <div className="text-xl font-bold font-mono text-white mt-0.5">
            {route.time_minutes.toFixed(1)} <span className="text-xs font-normal text-gray-400">min</span>
          </div>
        </div>

        <div className="bg-navy-900 p-3 rounded-lg border border-gray-800/80">
          <div className="text-[11px] text-gray-500 uppercase font-mono">Distance</div>
          <div className="text-xl font-bold font-mono text-white mt-0.5">
            {route.distance_km.toFixed(2)} <span className="text-xs font-normal text-gray-400">km</span>
          </div>
        </div>

        <div className="bg-navy-900 p-3 rounded-lg border border-gray-800/80">
          <div className="text-[11px] text-gray-500 uppercase font-mono">High-Risk Segs</div>
          <div
            className={`text-xl font-bold font-mono mt-0.5 ${
              (route.high_risk_segments_count || 0) > 1 ? 'text-red-400' : 'text-emerald-400'
            }`}
          >
            {route.high_risk_segments_count ?? 1}
          </div>
        </div>

        <div className="bg-navy-900 p-3 rounded-lg border border-gray-800/80">
          <div className="text-[11px] text-gray-500 uppercase font-mono">Route Cost</div>
          <div className="text-xl font-bold font-mono text-white mt-0.5">
            {route.route_cost ? route.route_cost.toFixed(1) : (route.time_minutes + 0.15 * route.overall_calculated_risk).toFixed(1)}
          </div>
        </div>
      </div>

      {/* Potholes & Conflicts Encountered */}
      <div className="grid grid-cols-2 gap-3 text-xs bg-navy-950/60 p-3 rounded-lg border border-gray-800 mb-4">
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Pothole Risk Exposure:</span>
          <span
            className={`font-semibold ${
              route.pothole_risk === 'HIGH' ? 'text-red-400' : route.pothole_risk === 'MEDIUM' ? 'text-amber-400' : 'text-emerald-400'
            }`}
          >
            {route.pothole_risk} ({route.hazards_count ?? (isFastest ? 4 : 0)} defects)
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-gray-400">Conflict Hotspots:</span>
          <span
            className={`font-semibold ${
              route.junction_risk === 'HIGH' ? 'text-red-400' : route.junction_risk === 'MEDIUM' ? 'text-amber-400' : 'text-emerald-400'
            }`}
          >
            {route.junction_risk} ({route.conflict_hotspots_count ?? (isFastest ? 2 : 0)} hotspots)
          </span>
        </div>
      </div>

      {/* Turn-by-Turn Maneuvers Preview */}
      {route.maneuvers && route.maneuvers.length > 0 && (
        <div className="mb-4 rounded-lg border border-gray-800/70 bg-navy-900/60 p-3 text-xs">
          <div className="text-[11px] font-mono uppercase text-gray-400 mb-2 flex items-center justify-between">
            <span>Route Maneuvers ({route.maneuvers.length} steps)</span>
            <span className="text-[10px] text-gray-500 font-normal">Deterministic geometry</span>
          </div>
          <div className="space-y-1.5">
            {route.maneuvers.map((m) => (
              <div key={m.step_index} className="flex items-start gap-2 text-gray-300">
                <span className="font-mono text-emerald-400 font-bold shrink-0">{getManeuverIcon(m.maneuver_type)}</span>
                <span className="leading-snug">{m.instruction}</span>
                <span className="ml-auto font-mono text-[10px] text-gray-500 shrink-0">
                  {Math.round(m.distance_meters)}m
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Action Footer */}
      <div className="flex items-center justify-between pt-2 border-t border-gray-800">
        <button
          onClick={(e) => {
            e.stopPropagation()
            onSelect()
          }}
          className={`text-xs font-semibold px-3 py-1.5 rounded transition ${
            isSelected ? 'text-emerald-400 bg-emerald-950/40 border border-emerald-700/60' : 'text-gray-400 hover:text-white'
          }`}
        >
          {isSelected ? '✓ Selected' : 'Select Option'}
        </button>

        <button
          onClick={(e) => {
            e.stopPropagation()
            onStartNavigation()
          }}
          className="text-xs font-bold px-4 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white transition flex items-center gap-1.5 shadow"
        >
          <span>🎙 Start Safe Navigation</span>
        </button>
      </div>
    </div>
  )
}

export default function SaferRoutes() {
  const [searchParams] = useSearchParams()
  const { data: roads = [] } = useRoads()
  const [originId, setOriginId] = useState<number | null>(1) // Default School Road
  const [destId, setDestId] = useState<number | null>(2) // Default Market Junction

  const [selectedRouteKey, setSelectedRouteKey] = useState<string>('safer')
  const [navState, setNavState] = useState<NavigationState | null>(null)
  const navStateRef = useRef<NavigationState | null>(null)

  const { speakNavigationClue, stopSpeaking } = useVoice()
  const [recitation, setRecitation] = useState<{ isReading: boolean; currentSpeechIndex: number; total: number }>({
    isReading: false,
    currentSpeechIndex: 0,
    total: 0,
  })
  const recitationCancelledRef = useRef<boolean>(false)
  const recitationTimeoutRef = useRef<any>(null)
  const [noticeMessage, setNoticeMessage] = useState<string | null>(null)

  useEffect(() => {
    navStateRef.current = navState
  }, [navState])

  const showNotice = useCallback((msg: string) => {
    setNoticeMessage(msg)
    setTimeout(() => {
      setNoticeMessage((prev) => (prev === msg ? null : prev))
    }, 3500)
  }, [])

  const handleStopRecitation = useCallback(() => {
    recitationCancelledRef.current = true
    if (recitationTimeoutRef.current) {
      clearTimeout(recitationTimeoutRef.current)
      recitationTimeoutRef.current = null
    }
    stopSpeaking()
    setRecitation((prev) => ({ ...prev, isReading: false }))
  }, [stopSpeaking])

  const handleReadDirections = useCallback((startIndex?: number) => {
    const current = navStateRef.current
    if (!current || current.status === 'IDLE') {
      showNotice('Start navigation first.')
      return
    }

    handleStopRecitation()
    recitationCancelledRef.current = false

    const steps = current.steps || []
    if (steps.length === 0) {
      setRecitation({ isReading: true, currentSpeechIndex: 0, total: 1 })
      speakNavigationClue(current.current_instruction, () => {
        setRecitation((prev) => ({ ...prev, isReading: false }))
      })
      return
    }

    const initialIdx = startIndex !== undefined
      ? Math.max(0, Math.min(startIndex, steps.length - 1))
      : (current.current_step || 0)

    setRecitation({
      isReading: true,
      currentSpeechIndex: initialIdx,
      total: steps.length,
    })

    const reciteStep = (idx: number) => {
      if (recitationCancelledRef.current) return
      if (idx >= steps.length) {
        setRecitation((prev) => ({ ...prev, isReading: false }))
        return
      }

      setRecitation({
        isReading: true,
        currentSpeechIndex: idx,
        total: steps.length,
      })

      const step = steps[idx]
      const clueText = step.hazard_warning
        ? `${step.instruction}. Caution: ${step.hazard_warning}`
        : step.instruction

      speakNavigationClue(clueText, () => {
        if (recitationCancelledRef.current) return
        recitationTimeoutRef.current = setTimeout(() => {
          if (recitationCancelledRef.current) return
          reciteStep(idx + 1)
        }, 350)
      })
    }

    reciteStep(initialIdx)
  }, [handleStopRecitation, speakNavigationClue, showNotice])

  const handleReadNext = useCallback(() => {
    const current = navStateRef.current
    if (!current || current.status === 'IDLE') {
      showNotice('Start navigation first.')
      return
    }
    handleStopRecitation()
    recitationCancelledRef.current = false
    const clueText = current.hazard_warning
      ? `${current.current_instruction}. Caution: ${current.hazard_warning}`
      : current.current_instruction
    speakNavigationClue(clueText)
  }, [handleStopRecitation, speakNavigationClue, showNotice])

  const { data: result, isLoading } = useRouteComparison(originId, destId)

  // Fetch navigation state from backend
  const refreshNavState = useCallback(async () => {
    try {
      const res = await fetch('/api/voice/navigation/state')
      if (res.ok) {
        const json = await res.json()
        if (json.success && json.navigation_state) {
          setNavState(json.navigation_state)
        }
      }
    } catch (e) {
      console.error('Error fetching voice navigation state:', e)
    }
  }, [])

  // Subscribe to VoiceActionDispatcher for FRIDAY actions
  useEffect(() => {
    refreshNavState()

    const unsubSelect = voiceActionDispatcher.subscribe('select_route', (payload) => {
      const key = payload.route_key || 'safer'
      setSelectedRouteKey(key)
      return true
    })

    const unsubStart = voiceActionDispatcher.subscribe('start_navigation', (payload) => {
      if (payload.navigation_state) {
        setNavState(payload.navigation_state)
      } else {
        refreshNavState()
      }
      if (payload.route_key) {
        setSelectedRouteKey(payload.route_key)
      }
      return true
    })

    const unsubPause = voiceActionDispatcher.subscribe('pause_navigation', () => {
      setNavState((prev) => (prev ? { ...prev, status: 'PAUSED' } : null))
      return true
    })

    const unsubResume = voiceActionDispatcher.subscribe('resume_navigation', () => {
      setNavState((prev) => (prev ? { ...prev, status: 'ACTIVE' } : null))
      return true
    })

    const unsubStop = voiceActionDispatcher.subscribe('stop_navigation', () => {
      setNavState((prev) => (prev ? { ...prev, status: 'IDLE' } : null))
      return true
    })

    const unsubReadDirections = voiceActionDispatcher.subscribe('read_navigation_directions', (payload) => {
      const current = navStateRef.current
      if (!current || current.status === 'IDLE') {
        showNotice('Start navigation first.')
        return true
      }
      const sIdx = payload.start_index !== undefined ? Number(payload.start_index) : undefined
      handleReadDirections(sIdx)
      return true
    })

    const unsubReadNext = voiceActionDispatcher.subscribe('read_next_navigation_direction', () => {
      const current = navStateRef.current
      if (!current || current.status === 'IDLE') {
        showNotice('Start navigation first.')
        return true
      }
      handleReadNext()
      return true
    })

    const unsubStopSpeech = voiceActionDispatcher.subscribe('stop_navigation_speech', () => {
      handleStopRecitation()
      return true
    })

    return () => {
      unsubSelect()
      unsubStart()
      unsubPause()
      unsubResume()
      unsubStop()
      unsubReadDirections()
      unsubReadNext()
      unsubStopSpeech()
      handleStopRecitation()
    }
  }, [refreshNavState, handleReadDirections, handleReadNext, handleStopRecitation, showNotice])

  // Handle URL params e.g. /routes?route=fastest or ?nav=active
  useEffect(() => {
    const routeParam = searchParams.get('route')
    if (routeParam) {
      setSelectedRouteKey(routeParam)
    }
    const navParam = searchParams.get('nav')
    if (navParam === 'active') {
      refreshNavState()
    }
  }, [searchParams, refreshNavState])

  // Navigation control handlers
  const handleStartNavigation = async (routeKey: string = selectedRouteKey) => {
    handleStopRecitation()
    try {
      const res = await fetch('/api/voice/navigation/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: 'default_session',
          route_key: routeKey,
          origin_road_id: originId ?? 1,
          destination_road_id: destId ?? 2,
        }),
      })
      if (res.ok) {
        const json = await res.json()
        if (json.navigation_state) {
          setNavState(json.navigation_state)
          setSelectedRouteKey(routeKey)
        }
      }
    } catch (e) {
      console.error('Error starting navigation:', e)
    }
  }

  const handlePauseNavigation = async () => {
    handleStopRecitation()
    try {
      await fetch('/api/voice/navigation/pause', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: 'default_session' }),
      })
      setNavState((prev) => (prev ? { ...prev, status: 'PAUSED' } : null))
    } catch (e) {
      console.error('Error pausing navigation:', e)
    }
  }

  const handleResumeNavigation = async () => {
    try {
      await fetch('/api/voice/navigation/resume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: 'default_session' }),
      })
      setNavState((prev) => (prev ? { ...prev, status: 'ACTIVE' } : null))
    } catch (e) {
      console.error('Error resuming navigation:', e)
    }
  }

  const handleStopNavigation = async () => {
    handleStopRecitation()
    try {
      await fetch('/api/voice/navigation/stop', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: 'default_session' }),
      })
      setNavState((prev) => (prev ? { ...prev, status: 'IDLE' } : null))
    } catch (e) {
      console.error('Error stopping navigation:', e)
    }
  }

  const handleAdvanceStep = async () => {
    handleStopRecitation()
    try {
      const res = await fetch('/api/voice/navigation/advance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: 'default_session' }),
      })
      if (res.ok) {
        refreshNavState()
      }
    } catch (e) {
      console.error('Error advancing step:', e)
    }
  }

  const fastestRoute = result?.routes.find((r) => r.route_key === 'fastest')
  const saferRoute = result?.routes.find((r) => r.route_key === 'safer')

  const timeDiff = fastestRoute && saferRoute
    ? Math.max(0.1, Number((saferRoute.time_minutes - fastestRoute.time_minutes).toFixed(1)))
    : 3.2

  const segsAvoided = fastestRoute && saferRoute
    ? Math.max(1, (fastestRoute.high_risk_segments_count || 3) - (saferRoute.high_risk_segments_count || 1))
    : 3

  const isNavActive = navState && (navState.status === 'ACTIVE' || navState.status === 'PAUSED')

  return (
    <Layout
      title="Route Intelligence & Voice-Guided Safer Navigation"
      subtitle="Compare Fastest Arterial Route vs Lower Calculated-Risk Route — Neutral Decision Support"
    >
      <DemoDataBanner />

      {/* Active Navigation Mode HUD */}
      {isNavActive && (
        <SafeNavigationHUD
          nav={navState}
          onPause={handlePauseNavigation}
          onResume={handleResumeNavigation}
          onStop={handleStopNavigation}
          onAdvance={handleAdvanceStep}
          recitation={recitation}
          onReadDirections={handleReadDirections}
          onReadNext={handleReadNext}
          onStopRecitation={handleStopRecitation}
        />
      )}

      {/* Non-blocking Notice Toast */}
      {noticeMessage && (
        <div className="fixed bottom-6 right-6 z-50 flex items-center gap-2 rounded-xl border border-amber-500/50 bg-navy-950/95 px-4 py-3 text-sm font-medium text-amber-200 shadow-2xl backdrop-blur">
          <span>⚠️</span>
          <span>{noticeMessage}</span>
        </div>
      )}

      {/* Origin & Destination Pickers */}
      <div className="mb-6 p-4 rounded-xl bg-navy-800 border border-gray-800 flex flex-wrap items-center gap-4 text-xs">
        <div className="flex items-center gap-2">
          <label htmlFor="origin-select" className="text-gray-400 font-semibold">Origin Point:</label>
          <select
            id="origin-select"
            value={originId ?? 1}
            onChange={(e) => setOriginId(Number(e.target.value))}
            className="bg-navy-900 border border-gray-700 text-white rounded-lg px-3 py-1.5 focus:outline-none focus:border-accent"
          >
            {roads.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name} (Risk: {Math.round(r.risk_score)})
              </option>
            ))}
          </select>
        </div>

        <div className="text-gray-500 font-bold">➔</div>

        <div className="flex items-center gap-2">
          <label htmlFor="dest-select" className="text-gray-400 font-semibold">Destination Point:</label>
          <select
            id="dest-select"
            value={destId ?? 2}
            onChange={(e) => setDestId(Number(e.target.value))}
            className="bg-navy-900 border border-gray-700 text-white rounded-lg px-3 py-1.5 focus:outline-none focus:border-accent"
          >
            {roads.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name} (Risk: {Math.round(r.risk_score)})
              </option>
            ))}
          </select>
        </div>

        <div className="ml-auto text-[11px] text-gray-400 font-mono bg-navy-900 px-3 py-1.5 rounded-lg border border-gray-800">
          Formula: <span className="text-accent font-semibold">RouteCost = TravelTime + 0.15 × RiskPenalty</span>
        </div>
      </div>

      {/* Comparative Explanation Card */}
      <div className="mb-6 p-4 rounded-xl bg-emerald-950/30 border border-emerald-700/50 flex items-start gap-3">
        <span className="text-2xl">💡</span>
        <div>
          <div className="text-sm font-bold text-emerald-300">
            Navigation Safety Trade-off Analysis
          </div>
          <p className="text-xs text-emerald-200 mt-1 leading-relaxed">
            &ldquo;The alternative route takes approximately <strong>{timeDiff} minutes longer</strong> and avoids{' '}
            <strong>{segsAvoided} high-risk segments</strong> based on available observations.&rdquo;
          </p>
          <div className="mt-2 text-[11px] text-gray-400 italic">
            ℹ️ The user retains final route choice. SafeCity Loop V2 presents neutral calculated risk trade-offs without coercive routing.
          </div>
        </div>
      </div>

      {/* Routes Comparison Grid: Fastest, Balanced, Safer */}
      {isLoading ? (
        <div className="h-64 flex items-center justify-center text-gray-400 text-xs">
          Computing route alternatives and surrogate safety penalties...
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {result?.routes && result.routes.length > 0 ? (
            result.routes.map((r) => (
              <RouteCard
                key={r.route_key}
                route={r}
                isSelected={selectedRouteKey === r.route_key}
                onSelect={() => setSelectedRouteKey(r.route_key)}
                onStartNavigation={() => handleStartNavigation(r.route_key)}
              />
            ))
          ) : (
            <div className="col-span-3 text-center py-12 text-gray-400 text-xs">
              No route options found between selected points.
            </div>
          )}
        </div>
      )}
    </Layout>
  )
}
