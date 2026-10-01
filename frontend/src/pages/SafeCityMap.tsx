import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import Layout from '@/components/layout/Layout'
import DemoDataBanner from '@/components/ui/DemoDataBanner'
import MapProvider from '@/components/map/MapProvider'
import RiskBadge from '@/components/ui/RiskBadge'
import StatusBadge from '@/components/ui/StatusBadge'
import { useHazards, useRoads, useDangerZones, useEvidence } from '@/hooks/useApi'
import { voiceActionDispatcher } from '@/services/voiceActionDispatcher'
import type { Hazard, Road, RoadSafetyEvaluation } from '@/types'

const HAZARD_TYPES = ['all', 'pothole', 'near-miss', 'dangerous-junction', 'road-work']

function scoreColor(score: number): string {
  if (score >= 80) return '#ef4444'
  if (score >= 55) return '#f97316'
  if (score >= 35) return '#f59e0b'
  return '#10b981'
}

export default function SafeCityMap() {
  const [searchParams] = useSearchParams()
  const [filter, setFilter] = useState('all')
  const [selectedRoadId, setSelectedRoadId] = useState<number | null>(1) // Default to School Road
  const [selectedHazard, setSelectedHazard] = useState<Hazard | null>(null)
  const [showRoads, setShowRoads] = useState(true)
  const [showHazards, setShowHazards] = useState(true)
  const [mapCenter, setMapCenter] = useState<[number, number]>([51.5074, -0.1278])
  const [mapZoom, setMapZoom] = useState<number>(14)

  const { data: hazards = [] } = useHazards()
  const { data: roads = [] } = useRoads()
  const { data: evaluations = [] } = useDangerZones()
  const { data: evidenceFiles = [] } = useEvidence(
    selectedRoadId ? { road_id: selectedRoadId } : undefined
  )

  const filteredHazards = filter === 'all'
    ? hazards
    : hazards.filter((h) => h.type === filter)

  const selectedRoad = roads.find((r) => r.id === selectedRoadId) || roads[0] || null

  const roadEvaluation: RoadSafetyEvaluation | undefined = selectedRoad
    ? evaluations.find((e) => e.road_id === selectedRoad.id)
    : undefined

  const roadHazards = selectedRoad
    ? hazards.filter((h) => h.road_id === selectedRoad.id)
    : []

  const potholes = roadHazards.filter((h) => h.type === 'pothole')
  const conflicts = roadHazards.filter((h) => h.type === 'near-miss' || h.type === 'dangerous-junction')

  // Coverage and Freshness calculation
  const coverageTier = roadEvaluation?.coverage || (
    (selectedRoad?.data_coverage || 80) >= 75 ? 'HIGH COVERAGE'
    : (selectedRoad?.data_coverage || 80) >= 45 ? 'MEDIUM COVERAGE'
    : 'LOW COVERAGE'
  )
  const isLowCoverage = coverageTier.includes('LOW') || (selectedRoad?.data_coverage || 80) < 50

  const handleSelectRoad = (road: Road) => {
    setSelectedRoadId(road.id)
    setSelectedHazard(null)
  }

  const handleMarkerClick = (hazard: Hazard) => {
    setSelectedHazard(hazard)
    if (hazard.road_id) {
      setSelectedRoadId(hazard.road_id)
    }
  }

  // Handle URL query parameters (e.g. ?lat=...&lng=... or ?hazard=...)
  useEffect(() => {
    const latParam = searchParams.get('lat')
    const lngParam = searchParams.get('lng')
    const zoomParam = searchParams.get('zoom')
    if (latParam && lngParam) {
      const lat = parseFloat(latParam)
      const lng = parseFloat(lngParam)
      const zoom = zoomParam ? parseInt(zoomParam, 10) : 15
      if (!isNaN(lat) && !isNaN(lng)) {
        setMapCenter([lat, lng])
        setMapZoom(zoom)
      }
    }

    const hazardParam = searchParams.get('hazard') || searchParams.get('highlight')
    if (hazardParam && hazards.length > 0) {
      const hId = parseInt(hazardParam, 10)
      const target = hazards.find((h) => h.id === hId)
      if (target) {
        setSelectedHazard(target)
        if (target.road_id) setSelectedRoadId(target.road_id)
        if (target.latitude && target.longitude) {
          setMapCenter([target.latitude, target.longitude])
          setMapZoom(16)
        }
      }
    }
  }, [searchParams, hazards])

  // Subscribe to live FRIDAY voice actions
  useEffect(() => {
    const unsubFocus = voiceActionDispatcher.subscribe('focus_map', (payload) => {
      const lat = Number(payload.lat)
      const lng = Number(payload.lng)
      const zoom = payload.zoom ? Number(payload.zoom) : 15
      if (!isNaN(lat) && !isNaN(lng)) {
        setMapCenter([lat, lng])
        setMapZoom(zoom)
        return true
      }
      return false
    })

    const unsubHighlightHazard = voiceActionDispatcher.subscribe('highlight_hazard', (payload) => {
      const hId = Number(payload.id)
      const target = hazards.find((h) => h.id === hId)
      if (target) {
        setSelectedHazard(target)
        if (target.road_id) setSelectedRoadId(target.road_id)
        if (target.latitude && target.longitude) {
          setMapCenter([target.latitude, target.longitude])
          setMapZoom(16)
        }
        return true
      }
      return false
    })

    const unsubHighlightMapEntity = voiceActionDispatcher.subscribe('highlight_map_entity', (payload) => {
      const hId = Number(payload.id)
      const target = hazards.find((h) => h.id === hId)
      if (target) {
        setSelectedHazard(target)
        if (target.road_id) setSelectedRoadId(target.road_id)
        return true
      }
      return false
    })

    const unsubDanger = voiceActionDispatcher.subscribe('show_danger_zones', () => {
      setShowRoads(true)
      setShowHazards(true)
      return true
    })

    return () => {
      unsubFocus()
      unsubHighlightHazard()
      unsubHighlightMapEntity()
      unsubDanger()
    }
  }, [hazards])

  return (
    <Layout title="SafeCity Risk Map" subtitle="Interactive multi-factor road risk, visual defect mapping, and coverage telemetry">
      <DemoDataBanner />

      {/* Layer Toggles & Map Controls */}
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 p-3 bg-navy-800 rounded-xl border border-gray-800 text-xs">
        <div className="flex items-center gap-3">
          <span className="font-semibold text-gray-300">Map Layers:</span>
          <label className="flex items-center gap-1.5 cursor-pointer text-gray-300">
            <input
              type="checkbox"
              checked={showRoads}
              onChange={(e) => setShowRoads(e.target.checked)}
              className="rounded bg-navy-900 border-gray-700 text-accent focus:ring-0"
            />
            <span>Road Segments & Risk Polylines</span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer text-gray-300">
            <input
              type="checkbox"
              checked={showHazards}
              onChange={(e) => setShowHazards(e.target.checked)}
              className="rounded bg-navy-900 border-gray-700 text-accent focus:ring-0"
            />
            <span>Potholes & Conflict Markers</span>
          </label>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-gray-400">Filter Incidents:</span>
          {HAZARD_TYPES.map((t) => (
            <button
              key={t}
              onClick={() => setFilter(t)}
              className={`px-2.5 py-1 rounded capitalize transition-all ${
                filter === t
                  ? 'bg-accent text-white font-semibold shadow'
                  : 'bg-navy-900 text-gray-400 hover:text-white border border-gray-800'
              }`}
            >
              {t === 'all' ? 'All' : t.replace(/-/g, ' ')}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 min-h-[620px]">
        {/* Map View (2 columns on large screens) */}
        <div className="lg:col-span-2 card p-2 flex flex-col">
          <div className="flex-1 rounded-xl overflow-hidden min-h-[500px]">
            <MapProvider
              hazards={filteredHazards}
              roads={roads}
              selectedRoadId={selectedRoadId}
              showRoads={showRoads}
              showHazards={showHazards}
              onRoadClick={handleSelectRoad}
              onMarkerClick={handleMarkerClick}
              center={mapCenter}
              zoom={mapZoom}
              className="h-full w-full"
            />
          </div>

          <div className="mt-2 px-2 py-1.5 flex flex-wrap items-center justify-between text-[11px] text-gray-400 border-t border-gray-800 font-mono">
            <span>🔴 Very High Risk (&gt;80) • 🟠 High (55–79) • 🟡 Moderate (35–54) • 🟢 Lower (&lt;35)</span>
            <span>NO DATA ≠ SAFE ROAD</span>
          </div>
        </div>

        {/* Right Drawer: Selected Road Segment Dossier */}
        <div className="card p-4 flex flex-col overflow-y-auto max-h-[700px] space-y-4">
          <div className="border-b border-gray-800 pb-3 flex items-start justify-between">
            <div>
              <span className="text-[10px] uppercase tracking-wider font-mono text-accent font-semibold">
                Corridor Telemetry Dossier
              </span>
              <h2 className="text-base font-bold text-white mt-0.5">
                {selectedRoad ? selectedRoad.name : 'Select a Road Segment'}
              </h2>
              <div className="text-xs text-gray-400 mt-0.5">
                Type: {selectedRoad?.road_type?.replace(/_/g, ' ') || 'Urban Arterial'}
              </div>
            </div>
            {selectedRoad && (
              <select
                value={selectedRoadId ?? ''}
                onChange={(e) => setSelectedRoadId(Number(e.target.value))}
                className="bg-navy-900 border border-gray-700 text-white rounded px-2 py-1 text-xs focus:outline-none focus:border-accent"
                aria-label="Select Road Corridor"
              >
                {roads.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.name}
                  </option>
                ))}
              </select>
            )}
          </div>

          {selectedRoad && (
            <>
              {/* Risk & SafeCity Scores */}
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-navy-900/90 p-3 rounded-lg border border-gray-800 text-center">
                  <div className="text-[11px] text-gray-400 uppercase font-mono">Calculated Risk</div>
                  <div
                    className="text-2xl font-black font-mono mt-1"
                    style={{ color: scoreColor(selectedRoad.risk_score) }}
                  >
                    {Math.round(selectedRoad.risk_score)}
                    <span className="text-xs text-gray-500 font-normal">/100</span>
                  </div>
                  <div className="text-[10px] text-gray-400 mt-0.5 font-semibold">
                    {selectedRoad.risk_score >= 80 ? 'VERY HIGH'
                      : selectedRoad.risk_score >= 55 ? 'HIGH'
                      : selectedRoad.risk_score >= 35 ? 'MODERATE' : 'LOWER'}
                  </div>
                </div>

                <div className="bg-navy-900/90 p-3 rounded-lg border border-gray-800 text-center">
                  <div className="text-[11px] text-gray-400 uppercase font-mono">SafeCity Score</div>
                  <div className="text-2xl font-black font-mono mt-1 text-emerald-400">
                    {Math.round(selectedRoad.safe_city_score)}
                    <span className="text-xs text-gray-500 font-normal">/100</span>
                  </div>
                  <div className="text-[10px] text-gray-400 mt-0.5 font-semibold">
                    Calculated Safety Level
                  </div>
                </div>
              </div>

              {/* Data Quality & Coverage Status (Requirement 4) */}
              <div className="bg-navy-900/70 p-3 rounded-lg border border-gray-800 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-400">Coverage Status:</span>
                  <span
                    className={`font-mono font-bold px-2 py-0.5 rounded text-[10px] border ${
                      coverageTier.includes('HIGH')
                        ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                        : coverageTier.includes('MEDIUM')
                        ? 'bg-amber-950 text-amber-400 border-amber-800'
                        : 'bg-red-950 text-red-400 border-red-800'
                    }`}
                  >
                    {coverageTier}
                  </span>
                </div>

                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-400">Model Confidence:</span>
                  <span className="text-white font-mono font-semibold">
                    {roadEvaluation?.confidence || 'HIGH'} ({Math.round((selectedRoad.risk_confidence || 0.85) * 100)}%)
                  </span>
                </div>

                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-400">Observation Freshness:</span>
                  <span className="text-gray-300 font-mono text-[11px]">
                    {selectedRoad.last_observed ? new Date(selectedRoad.last_observed).toLocaleString() : 'Within 24 Hours'}
                  </span>
                </div>

                {isLowCoverage && (
                  <div className="bg-amber-900/30 border border-amber-700/50 rounded p-2 text-xs text-amber-300">
                    ⚠️ <strong>Warning:</strong> Limited recent observations. Risk may be underestimated. NO DATA ≠ SAFE ROAD.
                  </div>
                )}
              </div>

              {/* Corridor Context & Exposure */}
              <div className="space-y-1.5 text-xs bg-navy-900/40 p-3 rounded-lg border border-gray-800/80">
                <div className="flex justify-between">
                  <span className="text-gray-400">Traffic Exposure:</span>
                  <span className="font-semibold text-white">{selectedRoad.traffic_exposure}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Pedestrian Vulnerability:</span>
                  <span className="font-semibold text-amber-400">{selectedRoad.vulnerability}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Repair Status:</span>
                  <span className="font-semibold text-sky-400 capitalize">{selectedRoad.status || 'Monitored'}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Main Contributing Factor:</span>
                  <span className="font-semibold text-red-400 truncate max-w-[180px]">
                    {roadEvaluation?.main_contributing_factor || 'Observed Surface Defects'}
                  </span>
                </div>
              </div>

              {/* Active Hazards on this Road */}
              <div>
                <div className="flex items-center justify-between text-xs font-semibold text-white mb-2">
                  <span>Active Defects &amp; Conflicts ({roadHazards.length})</span>
                  <span className="text-[11px] text-gray-400 font-mono">
                    {potholes.length} Potholes • {conflicts.length} Conflicts
                  </span>
                </div>

                {roadHazards.length === 0 ? (
                  <div className="p-3 bg-navy-900/60 rounded text-center text-xs text-gray-400">
                    No active defects registered on this segment.
                  </div>
                ) : (
                  <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                    {roadHazards.map((h) => (
                      <div
                        key={h.id}
                        onClick={() => setSelectedHazard(h)}
                        className={`p-2 rounded border text-xs cursor-pointer transition-colors ${
                          selectedHazard?.id === h.id
                            ? 'bg-accent/20 border-accent text-white'
                            : 'bg-navy-900 border-gray-800 text-gray-300 hover:border-gray-700'
                        }`}
                      >
                        <div className="flex items-center justify-between font-medium">
                          <span className="capitalize">{h.type.replace('-', ' ')}</span>
                          <span className="font-mono text-[11px] text-accent">
                            Risk {Math.round(h.risk_score)}
                          </span>
                        </div>
                        <div className="flex items-center justify-between text-[10px] text-gray-400 mt-1 font-mono">
                          <span>Sev: {h.severity}</span>
                          {h.evidence_code && <span className="text-sky-400">{h.evidence_code}</span>}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Evidence Records */}
              {evidenceFiles.length > 0 && (
                <div className="border-t border-gray-800 pt-3">
                  <div className="text-xs font-semibold text-white mb-2">
                    Linked Visual Evidence ({evidenceFiles.length})
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {evidenceFiles.map((ev) => (
                      <span
                        key={ev.id}
                        className="bg-navy-900 text-sky-400 border border-sky-800/50 px-2 py-0.5 rounded text-[10px] font-mono"
                      >
                        {ev.evidence_code} ({ev.file_type})
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </Layout>
  )
}
