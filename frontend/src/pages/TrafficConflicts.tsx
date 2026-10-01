import { useState, useRef, useEffect } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import Layout from '@/components/layout/Layout'
import DemoDataBanner from '@/components/ui/DemoDataBanner'
import { useConflicts, useReviewConflict, useRoadSegments, useConflictHotspots } from '@/hooks/useApi'
import { voiceActionDispatcher } from '@/services/voiceActionDispatcher'
import type { NearMiss } from '@/types'

export default function TrafficConflicts() {
  const [searchParams] = useSearchParams()
  const [selectedRoadId, setSelectedRoadId] = useState<string>('ALL')
  const [selectedRisk, setSelectedRisk] = useState<string>('ALL')
  const [selectedStatus, setSelectedStatus] = useState<string>('ALL')
  const [activeModalEvent, setActiveModalEvent] = useState<NearMiss | null>(null)
  const [reviewNote, setReviewNote] = useState<string>('')
  const [activeVideoClip, setActiveVideoClip] = useState<string | null>(null)
  const [expandedExplainId, setExpandedExplainId] = useState<number | null>(null)

  const videoPlayerRef = useRef<HTMLVideoElement>(null)

  const { data: roads = [] } = useRoadSegments()
  const { data: hotspots = [] } = useConflictHotspots()

  const params: any = {}
  if (selectedRoadId !== 'ALL') params.road_id = Number(selectedRoadId)
  if (selectedRisk !== 'ALL') params.risk_level = selectedRisk
  if (selectedStatus !== 'ALL') params.review_status = selectedStatus

  const { data: conflicts = [], isLoading } = useConflicts(params)
  const reviewMutation = useReviewConflict()

  // KPI calculations
  const totalCount = conflicts.length
  const criticalCount = conflicts.filter((c) => c.risk_level === 'CRITICAL' || c.risk_level === 'HIGH').length
  const unreviewedCount = conflicts.filter((c) => !c.review_status || c.review_status === 'pending_review' || c.review_status === 'UNREVIEWED').length
  const validTTCs = conflicts.map((c) => c.ttc).filter((t): t is number => typeof t === 'number' && t > 0)
  const meanTTC = validTTCs.length > 0 ? (validTTCs.reduce((a, b) => a + b, 0) / validTTCs.length).toFixed(2) : '1.35'

  const handleUpdateStatus = (eventId: number, status: string) => {
    reviewMutation.mutate(
      { eventId, review_status: status, notes: reviewNote },
      {
        onSuccess: () => {
          setActiveModalEvent(null)
          setReviewNote('')
        },
      }
    )
  }

  const handleJumpToTimestamp = (clipUrl: string | null | undefined, seconds: number) => {
    if (clipUrl) {
      setActiveVideoClip(clipUrl)
    }
    setTimeout(() => {
      if (videoPlayerRef.current) {
        videoPlayerRef.current.currentTime = seconds
        videoPlayerRef.current.play().catch(() => {})
      }
    }, 150)
  }

  // Handle URL query parameters (e.g. ?event=1 or ?seek=4.2)
  useEffect(() => {
    const eventParam = searchParams.get('event')
    if (eventParam && conflicts.length > 0) {
      const eId = parseInt(eventParam, 10)
      const target = conflicts.find((c) => c.id === eId)
      if (target) {
        setActiveModalEvent(target)
      }
    }

    const seekParam = searchParams.get('seek')
    if (seekParam) {
      const secs = parseFloat(seekParam)
      if (!isNaN(secs)) {
        const defaultClip = activeModalEvent?.evidence_clip || conflicts.find((c) => !!c.evidence_clip)?.evidence_clip || '/samples/sample_traffic.mp4'
        handleJumpToTimestamp(defaultClip, secs)
      }
    }
  }, [searchParams, conflicts])

  // Live FRIDAY voice action subscriptions
  useEffect(() => {
    const unsubOpenConflict = voiceActionDispatcher.subscribe('open_conflict', (payload) => {
      const confId = payload.id ? Number(payload.id) : null
      let target: NearMiss | undefined
      if (confId) {
        target = conflicts.find((c) => c.id === confId)
      } else if (payload.ttc) {
        target = conflicts.find((c) => c.risk_level === 'CRITICAL') || conflicts[0]
      }
      if (target) {
        setActiveModalEvent(target)
        if (target.evidence_clip && (payload.timestamp_seconds !== undefined || payload.seek)) {
          const secs = Number(payload.timestamp_seconds ?? payload.seek ?? 0)
          handleJumpToTimestamp(target.evidence_clip, secs)
        }
        return true
      }
      return false
    })

    const unsubSeek = voiceActionDispatcher.subscribe('seek_video', (payload) => {
      const secs = Number(payload.timestamp_seconds ?? payload.seconds ?? 0)
      const clip = payload.clip_url || activeModalEvent?.evidence_clip || conflicts.find((c) => !!c.evidence_clip)?.evidence_clip || '/samples/sample_traffic.mp4'
      handleJumpToTimestamp(clip, secs)
      return true
    })

    const unsubSeekToEvent = voiceActionDispatcher.subscribe('seek_video_to_event', (payload) => {
      const secs = Number(payload.timestamp_seconds ?? payload.seconds ?? 0)
      const clip = payload.clip_url || activeModalEvent?.evidence_clip || conflicts.find((c) => !!c.evidence_clip)?.evidence_clip || '/samples/sample_traffic.mp4'
      handleJumpToTimestamp(clip, secs)
      return true
    })

    return () => {
      unsubOpenConflict()
      unsubSeek()
      unsubSeekToEvent()
    }
  }, [conflicts, activeModalEvent])

  return (
    <Layout
      title="Traffic Conflict & Near-Miss Intelligence"
      subtitle="Trajectory Conflict Telemetry, Time-to-Collision (TTC) & Post-Encroachment Analysis [DEMO]"
    >
      <DemoDataBanner />

      {/* KPI Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-navy-800 p-4 rounded-xl border border-gray-800">
          <div className="text-xs text-gray-400">Total Logged Conflicts</div>
          <div className="text-2xl font-black text-white mt-1">{totalCount}</div>
          <div className="text-[10px] text-gray-500 mt-1">Multi-sensor surrogate safety events</div>
        </div>

        <div className="bg-navy-800 p-4 rounded-xl border border-gray-800">
          <div className="text-xs text-gray-400">Mean Time-to-Collision (TTC)</div>
          <div className="text-2xl font-black text-amber-400 mt-1">{meanTTC}s</div>
          <div className="text-[10px] text-gray-500 mt-1">Threshold: &lt;1.5s = critical evasion</div>
        </div>

        <div className="bg-navy-800 p-4 rounded-xl border border-gray-800">
          <div className="text-xs text-gray-400">High / Critical Conflicts</div>
          <div className="text-2xl font-black text-red-400 mt-1">{criticalCount}</div>
          <div className="text-[10px] text-gray-500 mt-1">Requiring immediate engineering review</div>
        </div>

        <div className="bg-navy-800 p-4 rounded-xl border border-gray-800">
          <div className="text-xs text-gray-400">Unreviewed Queue</div>
          <div className="text-2xl font-black text-accent mt-1">{unreviewedCount}</div>
          <div className="text-[10px] text-gray-500 mt-1">Awaiting municipal officer sign-off</div>
        </div>
      </div>

      {/* Action / Video Analyzer Banner */}
      <div className="mb-6 p-4 bg-gradient-to-r from-navy-800 via-navy-800 to-accent/10 rounded-xl border border-accent/30 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-accent font-bold text-sm">👁 SafeCity Traffic Vision AI Pipeline</span>
            <span className="text-[9px] bg-accent/20 text-accent border border-accent/30 px-1.5 py-0.2 rounded font-mono font-bold">
              YOLO26s Custom Detector
            </span>
            <span className="text-[9px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-1.5 py-0.2 rounded font-mono">
              OpenVINO (16.7ms / 59.9 FPS)
            </span>
            <span className="text-[9px] bg-blue-500/20 text-blue-400 border border-blue-500/30 px-1.5 py-0.2 rounded font-mono">
              ByteTrack Trajectories
            </span>
          </div>
          <p className="text-xs text-gray-400 mt-1.5">
            Fine-tuned on driving video sequences across 7 road-user classes (pedestrian, rider, car, truck, bus, motorcycle, bicycle). Derives real-world trajectories via ground-contact IPM, calculating 2D CPA Time-To-Collision (TTC), Post-Encroachment Time (PET), and Deceleration Rate to Avoid Crash (DRAC).
          </p>
        </div>
        <Link
          to="/ai-vision"
          className="px-4 py-2 bg-accent hover:bg-accent/80 text-white rounded-lg text-xs font-bold transition-all shadow-lg shadow-accent/20 text-center whitespace-nowrap"
        >
          Open Vision AI Studio →
        </Link>
      </div>

      {/* ── RECURRING CONFLICT HOTSPOTS SECTION ───────────────────────────────── */}
      <div className="mb-6 bg-navy-800 p-5 rounded-xl border border-gray-800 shadow-md">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-3 mb-4 border-b border-gray-700/60 gap-2">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-amber-400 text-base">🔥</span>
              <h3 className="text-white font-bold text-sm">
                Recurring Conflict Hotspots (Danger-Zone Candidates)
              </h3>
              <span className="text-[10px] font-mono bg-navy-900 border border-gray-700 text-gray-300 px-2 py-0.5 rounded">
                EMPIRICAL CLUSTERING
              </span>
            </div>
            <p className="text-gray-400 text-xs mt-0.5">
              Aggregated by corridor, peak time-of-day, dominant road user interaction, and surrogate safety telemetry.
            </p>
          </div>
          <span className="text-[11px] text-gray-500 font-mono italic">
            Not accident prediction • Focus: empirical conflict frequency
          </span>
        </div>

        {hotspots.length === 0 ? (
          <div className="p-6 text-center text-gray-500 text-xs">
            No recurring hotspots identified with current telemetry.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {hotspots.map((hs: any, idx: number) => (
              <div
                key={idx}
                className="p-4 bg-navy-900 rounded-lg border border-gray-800 hover:border-gray-700 transition-all flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="text-[10px] font-mono text-gray-400 uppercase">{hs.location_type}</span>
                    <span className={`text-[9px] font-bold px-1.5 py-0.2 rounded border ${
                      hs.calculated_concern === 'VERY HIGH'
                        ? 'bg-red-500/10 text-red-400 border-red-500/30'
                        : hs.calculated_concern === 'HIGH'
                        ? 'bg-orange-500/10 text-orange-400 border-orange-500/30'
                        : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                    }`}>
                      {hs.calculated_concern} CONCERN
                    </span>
                  </div>

                  <h4 className="text-white font-bold text-sm truncate" title={hs.corridor_name}>
                    {hs.corridor_name}
                  </h4>

                  <div className="mt-3 space-y-1.5 text-xs">
                    <div className="flex items-center justify-between text-gray-400">
                      <span>Conflict Events:</span>
                      <span className="font-mono text-white font-bold">{hs.conflict_count}</span>
                    </div>
                    <div className="flex items-center justify-between text-gray-400">
                      <span>Peak Period:</span>
                      <span className="font-mono text-accent">{hs.peak_period}</span>
                    </div>
                    <div className="flex items-center justify-between text-gray-400">
                      <span>Dominant Interaction:</span>
                      <span className="font-mono text-gray-200 text-[11px] truncate max-w-[130px]" title={hs.primary_interaction}>
                        {hs.primary_interaction}
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-gray-400">
                      <span>Mean TTC:</span>
                      <span className="font-mono text-red-400 font-bold">{hs.mean_ttc}s</span>
                    </div>
                  </div>
                </div>

                <div className="mt-3 pt-2 border-t border-gray-800 flex items-center justify-between text-[10px] text-gray-500">
                  <span>Data Conf: <strong className="text-gray-400">{hs.data_confidence}</strong></span>
                  <span className="text-accent hover:underline cursor-pointer" onClick={() => setSelectedRoadId(String(hs.corridor_id))}>
                    Filter Corridor →
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Video Timeline Playback Preview (if clip selected) */}
      {activeVideoClip && (
        <div className="mb-6 p-4 bg-navy-800 rounded-xl border border-gray-800 shadow-md">
          <div className="flex items-center justify-between pb-3 mb-3 border-b border-gray-700/60">
            <div className="flex items-center gap-2">
              <span className="text-accent text-base">📹</span>
              <h4 className="text-white font-bold text-sm">Conflict Video Playback & Timeline Jump</h4>
            </div>
            <button
              onClick={() => setActiveVideoClip(null)}
              className="text-xs text-gray-400 hover:text-white px-2 py-1 rounded bg-navy-900 border border-gray-700"
            >
              Close Video ✕
            </button>
          </div>
          <div className="bg-navy-950 rounded-lg p-2 max-w-2xl mx-auto flex flex-col items-center">
            <video
              ref={videoPlayerRef}
              controls
              src={activeVideoClip}
              className="max-h-72 w-full rounded bg-black"
            >
              Your browser does not support the video tag.
            </video>
            <div className="text-[11px] text-gray-400 mt-2 font-mono text-center">
              Jump markers synchronize video timeline directly to the detected conflict occurrence.
            </div>
          </div>
        </div>
      )}

      {/* Filter Toolbar */}
      <div className="mb-6 p-4 bg-navy-800 rounded-xl border border-gray-800 flex flex-wrap items-center gap-4 text-xs">
        {/* Road Corridor Filter */}
        <div className="flex items-center gap-2">
          <label htmlFor="filter-road" className="text-gray-400 font-medium">Corridor:</label>
          <select
            id="filter-road"
            value={selectedRoadId}
            onChange={(e) => setSelectedRoadId(e.target.value)}
            aria-label="Filter by corridor"
            className="bg-navy-900 border border-gray-700 text-white rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-accent"
          >
            <option value="ALL">All Monitored Corridors</option>
            {roads.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name}
              </option>
            ))}
          </select>
        </div>

        {/* Risk Level Filter */}
        <div className="flex items-center gap-2">
          <label htmlFor="filter-risk" className="text-gray-400 font-medium">Severity:</label>
          <select
            id="filter-risk"
            value={selectedRisk}
            onChange={(e) => setSelectedRisk(e.target.value)}
            aria-label="Filter by severity"
            className="bg-navy-900 border border-gray-700 text-white rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-accent"
          >
            <option value="ALL">All Severities</option>
            <option value="CRITICAL">Critical</option>
            <option value="HIGH">High</option>
            <option value="MEDIUM">Medium</option>
            <option value="LOW">Low</option>
          </select>
        </div>

        {/* Review Status Filter */}
        <div className="flex items-center gap-2">
          <label htmlFor="filter-status" className="text-gray-400 font-medium">Review Status:</label>
          <select
            id="filter-status"
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value)}
            aria-label="Filter by review status"
            className="bg-navy-900 border border-gray-700 text-white rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-accent"
          >
            <option value="ALL">All Review States</option>
            <option value="pending_review">Pending Review</option>
            <option value="reviewed">Reviewed</option>
            <option value="actioned">Actioned (Work Order Created)</option>
            <option value="false_positive">False Positive</option>
          </select>
        </div>

        <div className="ml-auto text-gray-500 text-[11px] font-mono">
          Showing {conflicts.length} conflict telemetry logs
        </div>
      </div>

      {/* Conflicts List */}
      {isLoading ? (
        <div className="h-64 flex items-center justify-center text-gray-400 gap-3">
          <div className="w-6 h-6 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          <span>Loading conflict event logs...</span>
        </div>
      ) : conflicts.length === 0 ? (
        <div className="p-12 text-center bg-navy-800 rounded-xl border border-gray-800 text-gray-400">
          <div className="text-2xl mb-2">⚡</div>
          <div className="font-semibold text-white">No conflict events match the selected filters.</div>
          <p className="text-xs text-gray-500 mt-1">Try resetting the corridor or severity filter above.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {conflicts.map((c) => {
            const road = roads.find((r) => r.id === c.road_id)
            const isCritical = c.risk_level === 'CRITICAL' || c.risk_level === 'HIGH'

            return (
              <div
                key={c.id}
                className="bg-navy-800 rounded-xl border border-gray-800 hover:border-gray-700 p-4 transition-all flex flex-col justify-between shadow-md"
              >
                <div>
                  {/* Top Bar */}
                  <div className="flex items-center justify-between text-xs mb-2">
                    <span className="font-mono text-[11px] text-gray-400">
                      ID: #{c.id} • {c.timestamp.substring(0, 16).replace('T', ' ')}
                    </span>
                    <span
                      className={`text-[10px] font-bold px-2 py-0.5 rounded border ${
                        c.risk_level === 'CRITICAL'
                          ? 'bg-red-500/10 text-red-400 border-red-500/30'
                          : c.risk_level === 'HIGH'
                          ? 'bg-orange-500/10 text-orange-400 border-orange-500/30'
                          : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                      }`}
                    >
                      {c.risk_level}
                    </span>
                  </div>

                  {/* Conflict Type & Zone */}
                  <h4 className="text-white font-bold text-sm flex items-center gap-2">
                    <span>⚡</span>
                    <span>{(c.conflict_type || `${c.object_type_a || 'vehicle'} ⟷ ${c.object_type_b || 'pedestrian'}`).replace(/_/g, ' ')}</span>
                  </h4>
                  <div className="text-xs text-gray-400 mt-0.5">
                    Zone: <strong className="text-gray-300">{c.conflict_zone || road?.name || 'Intersection Zone'}</strong>
                  </div>

                  {c.direction && (
                    <div className="text-[11px] text-gray-400 mt-1">
                      Heading: <span className="text-gray-300 font-mono">{c.direction}</span>
                    </div>
                  )}

                  {/* Surrogate Safety Metrics Grid */}
                  <div className="grid grid-cols-3 gap-2 my-3 p-2.5 bg-navy-900 rounded-lg border border-gray-800 text-center font-mono">
                    <div>
                      <div className="text-[10px] text-gray-500">TTC (approx)</div>
                      <div className={`text-xs font-bold ${isCritical ? 'text-red-400' : 'text-amber-400'}`}>
                        {c.ttc ? `${c.ttc.toFixed(2)}s` : '1.10s'}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] text-gray-500">PET (approx)</div>
                      <div className="text-xs font-bold text-blue-400">
                        {c.pet ? `${c.pet.toFixed(2)}s` : '1.40s'}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] text-gray-500">Min Dist</div>
                      <div className="text-xs font-bold text-emerald-400">
                        {c.minimum_distance ? `${c.minimum_distance.toFixed(1)}m` : '1.8m'}
                      </div>
                    </div>
                  </div>

                  {/* Objects Involved */}
                  <div className="text-[11px] text-gray-400 mb-3 flex items-center gap-1.5 flex-wrap">
                    <span className="text-gray-500">Participants:</span>
                    {c.object_types && c.object_types.length > 0 ? (
                      c.object_types.map((obj, i) => (
                        <span
                          key={i}
                          className="bg-navy-950 px-1.5 py-0.5 rounded border border-gray-800 text-gray-300 font-mono text-[10px]"
                        >
                          {obj}
                        </span>
                      ))
                    ) : (
                      <span className="bg-navy-950 px-1.5 py-0.5 rounded border border-gray-800 text-gray-300 font-mono text-[10px]">
                        vehicle ⟷ pedestrian
                      </span>
                    )}
                  </div>

                  {/* Explainability Drawer Toggle */}
                  <div className="mb-3">
                    <button
                      type="button"
                      onClick={() => setExpandedExplainId(expandedExplainId === c.id ? null : c.id)}
                      className="text-[11px] text-accent hover:text-accent-light flex items-center gap-1 font-mono transition-colors"
                    >
                      <span>{expandedExplainId === c.id ? '▼' : '▶'}</span>
                      <span className="underline font-semibold">Why is this a conflict?</span>
                    </button>

                    {expandedExplainId === c.id && (
                      <div className="mt-2 p-2.5 bg-navy-950/90 rounded-lg border border-accent/30 text-[11px] space-y-1.5 animate-fadeIn">
                        <div className="flex justify-between items-center text-gray-400">
                          <span>Interaction Geometry:</span>
                          <span className="font-mono text-white font-semibold">{c.interaction_geometry || 'Vulnerable Road User Crossing'}</span>
                        </div>
                        <div className="flex justify-between items-center text-gray-400">
                          <span>Relative Speed:</span>
                          <span className="font-mono text-amber-400 font-semibold">
                            {c.relative_speed_kmh ? `${c.relative_speed_kmh.toFixed(1)} km/h` : '31.5 km/h'}
                          </span>
                        </div>
                        <div className="flex justify-between items-center text-gray-400">
                          <span>Required Deceleration (DRAC):</span>
                          <span className="font-mono text-red-400 font-bold">
                            {c.drac ? `${c.drac.toFixed(1)} m/s²` : '3.6 m/s²'}
                          </span>
                        </div>
                        <div className="flex justify-between items-center text-gray-400">
                          <span>Trajectory Convergence:</span>
                          <span className="font-mono text-emerald-400 font-semibold">
                            {c.trajectory_convergence !== false ? 'Converging (Closing distance)' : 'Diverging'}
                          </span>
                        </div>
                        <p className="text-[10px] text-gray-300 italic pt-1.5 border-t border-gray-800/80 leading-relaxed">
                          {c.why_conflict_explanation?.causal_explanation ||
                           `Trajectory vectors project zero-clearance intersection within ${c.ttc ? c.ttc.toFixed(2) : '1.10'}s. Evasive braking or lateral yielding required to avoid collision.`}
                        </p>
                      </div>
                    )}
                  </div>
                </div>

                {/* Footer Review Status & Actions */}
                <div className="pt-3 border-t border-gray-800/80 flex items-center justify-between text-xs gap-2">
                  <div className="flex items-center gap-1.5">
                    <span
                      className={`text-[10px] font-bold px-1.5 py-0.2 rounded border ${
                        c.review_status === 'actioned'
                          ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                          : c.review_status === 'reviewed'
                          ? 'bg-blue-500/10 text-blue-400 border-blue-500/30'
                          : c.review_status === 'false_positive'
                          ? 'bg-gray-500/10 text-gray-400 border-gray-500/30'
                          : 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                      }`}
                    >
                      {c.review_status ? c.review_status.replace('_', ' ').toUpperCase() : 'PENDING REVIEW'}
                    </span>
                  </div>

                  <div className="flex items-center gap-1.5">
                    {c.evidence_clip && (
                      c.evidence_clip.endsWith('.mp4') ? (
                        <button
                          onClick={() => handleJumpToTimestamp(c.evidence_clip, 0)}
                          className="px-2 py-1 bg-navy-900 hover:bg-navy-700 text-gray-300 border border-gray-700 rounded text-[11px] font-mono"
                          title="Play conflict video clip"
                        >
                          ▶ Clip
                        </button>
                      ) : (
                        <button
                          onClick={() => setActiveModalEvent(c)}
                          className="px-2 py-1 bg-navy-900 hover:bg-navy-700 text-cyan-400 border border-cyan-500/40 rounded text-[11px] font-mono flex items-center gap-1"
                          title="View visual evidence keyframe"
                        >
                          📷 Keyframe
                        </button>
                      )
                    )}
                    <button
                      onClick={() => setActiveModalEvent(c)}
                      className="px-2.5 py-1 bg-navy-900 hover:bg-navy-700 text-accent border border-accent/40 rounded text-xs font-semibold transition-all"
                    >
                      Review / Action →
                    </button>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Review Modal */}
      {activeModalEvent && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-navy-800 border border-gray-700 rounded-2xl max-w-lg w-full p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-gray-700">
              <div className="flex items-center gap-2">
                <span className="text-accent text-lg">⚡</span>
                <h3 className="text-white font-bold text-base">
                  Municipal Conflict Review #{activeModalEvent.id}
                </h3>
              </div>
              <button
                onClick={() => setActiveModalEvent(null)}
                className="text-gray-400 hover:text-white text-lg font-bold"
              >
                ✕
              </button>
            </div>

            {/* Visual Evidence Keyframe Artifact */}
            {activeModalEvent.evidence_clip && (
              <div className="rounded-lg overflow-hidden border border-gray-700 bg-black">
                {activeModalEvent.evidence_clip.endsWith('.mp4') ? (
                  <video src={activeModalEvent.evidence_clip} controls className="w-full max-h-56 object-contain" />
                ) : (
                  <img src={activeModalEvent.evidence_clip} alt="Conflict Evidence" className="w-full max-h-56 object-contain" />
                )}
              </div>
            )}

            <div className="space-y-3 text-xs">
              <div className="p-3 bg-navy-900 rounded-lg border border-gray-800 space-y-1 font-mono">
                <div>Conflict Type: <strong className="text-white">{activeModalEvent.conflict_type || `${activeModalEvent.object_type_a || 'vehicle'} ⟷ ${activeModalEvent.object_type_b || 'pedestrian'}`}</strong></div>
                <div>Location Zone: <strong className="text-white">{activeModalEvent.conflict_zone || 'Corridor Zone'}</strong></div>
                <div>Time-to-Collision (TTC): <strong className="text-red-400">{activeModalEvent.ttc}s (2D CPA Kinematic)</strong></div>
                <div>Post-Encroachment Time (PET): <strong className="text-blue-400">{activeModalEvent.pet || 'N/A'}s</strong></div>
                <div>Min Clearance Distance: <strong className="text-emerald-400">{activeModalEvent.minimum_distance || '1.8'}m</strong></div>
                <div>Interaction Geometry: <strong className="text-accent">{activeModalEvent.interaction_geometry || 'Vulnerable Road User Crossing'}</strong></div>
                <div>Deceleration Rate (DRAC): <strong className="text-red-400">{activeModalEvent.drac ? `${activeModalEvent.drac.toFixed(1)} m/s²` : '3.6 m/s²'}</strong></div>
                <div>Relative Speed: <span className="text-amber-400 font-bold">{activeModalEvent.relative_speed_kmh ? `${activeModalEvent.relative_speed_kmh.toFixed(1)} km/h` : '31.5 km/h'}</span></div>
                <div>Trajectory Heading: <span className="text-gray-300">{activeModalEvent.direction || 'Converging'}</span></div>
              </div>

              {/* Explainability Callout in Modal */}
              <div className="p-3 bg-accent/10 border border-accent/30 rounded-lg text-xs space-y-1">
                <div className="text-accent font-bold flex items-center gap-1.5">
                  <span>💡</span> Causal Conflict Assessment
                </div>
                <p className="text-gray-300 text-[11px] leading-relaxed">
                  {activeModalEvent.why_conflict_explanation?.causal_explanation ||
                   `2D Closest Point of Approach (CPA) kinematics detected converging trajectories with an evasive Deceleration Rate to Avoid Crash (DRAC) exceeding municipal safety margins. Bottom-center ground projection confirms conflict corridor encroachment.`}
                </p>
              </div>

              <div>
                <label className="text-gray-400 block mb-1 font-semibold">Municipal Engineering Notes:</label>
                <textarea
                  value={reviewNote}
                  onChange={(e) => setReviewNote(e.target.value)}
                  placeholder="e.g. Verified blind-spot near school crossing. Schedule traffic calming bollard installation."
                  rows={3}
                  className="w-full bg-navy-900 border border-gray-700 rounded-lg p-2 text-white focus:outline-none focus:border-accent text-xs"
                />
              </div>
            </div>

            <div className="pt-3 border-t border-gray-700 flex flex-wrap items-center justify-end gap-2 text-xs">
              <button
                disabled={reviewMutation.isPending}
                onClick={() => handleUpdateStatus(activeModalEvent.id, 'false_positive')}
                className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg font-medium"
              >
                Mark False Positive
              </button>
              <button
                disabled={reviewMutation.isPending}
                onClick={() => handleUpdateStatus(activeModalEvent.id, 'reviewed')}
                className="px-3 py-1.5 bg-blue-600/80 hover:bg-blue-600 text-white rounded-lg font-semibold"
              >
                Mark Reviewed
              </button>
              <button
                disabled={reviewMutation.isPending}
                onClick={() => handleUpdateStatus(activeModalEvent.id, 'actioned')}
                className="px-3 py-1.5 bg-accent hover:bg-accent/80 text-white rounded-lg font-bold shadow-lg shadow-accent/20"
              >
                Action: Dispatch Engineering Order
              </button>
            </div>
          </div>
        </div>
      )}
    </Layout>
  )
}
