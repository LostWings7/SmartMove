import { useEffect } from 'react'
import { MapContainer, TileLayer, CircleMarker, Polyline, Popup, Tooltip, useMap } from 'react-leaflet'
import type { MapProviderProps } from './MapProvider'
import type { Hazard, Road } from '@/types'

function ChangeView({ center, zoom }: { center: [number, number]; zoom: number }) {
  const map = useMap()
  useEffect(() => {
    map.setView(center, zoom)
  }, [center, zoom, map])
  return null
}

const TILE_URL = import.meta.env.VITE_TILE_URL ?? 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'
const TILE_ATTRIBUTION =
  import.meta.env.VITE_TILE_ATTRIBUTION ??
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'

const hazardColors: Record<string, string> = {
  pothole: '#ef4444',
  'near-miss': '#f97316',
  'dangerous-junction': '#ef4444',
  'road-work': '#f59e0b',
  'high-risk-zone': '#dc2626',
}

const hazardLabels: Record<string, string> = {
  pothole: '● Pothole Defect',
  'near-miss': '⚠ Traffic Conflict Candidate',
  'dangerous-junction': '⚠ Conflict Junction Hotspot',
  'road-work': '🔧 Municipal Work Zone',
  'high-risk-zone': '🔴 High-Risk Danger Corridor',
}

function getRiskColor(score: number): string {
  if (score >= 80) return '#ef4444' // Red
  if (score >= 55) return '#f97316' // Orange
  if (score >= 35) return '#f59e0b' // Amber
  return '#10b981' // Green
}

export default function LeafletMap({
  hazards = [],
  roads = [],
  selectedRoadId = null,
  center = [51.5074, -0.1278],
  zoom = 13,
  onMarkerClick,
  onRoadClick,
  showRoads = true,
  showHazards = true,
  className = '',
}: MapProviderProps) {
  return (
    <div className={`relative ${className}`} style={{ minHeight: 400 }}>
      <MapContainer
        center={center}
        zoom={zoom}
        style={{ height: '100%', width: '100%', minHeight: 400, background: '#0f172a' }}
        className="rounded-xl overflow-hidden"
      >
        <TileLayer
          url={TILE_URL}
          attribution={TILE_ATTRIBUTION}
          className="map-tiles"
        />
        <ChangeView center={center} zoom={zoom} />

        {/* 1. Road Segments Polyline Layer */}
        {showRoads &&
          roads.map((road) => {
            const isSelected = selectedRoadId === road.id
            const color = getRiskColor(road.risk_score)
            const coords = road.geometry && road.geometry.length > 0
              ? road.geometry
              : [
                  [road.latitude - 0.002, road.longitude - 0.002],
                  [road.latitude, road.longitude],
                  [road.latitude + 0.002, road.longitude + 0.002],
                ]

            return (
              <Polyline
                key={`road-${road.id}`}
                positions={coords as [number, number][]}
                pathOptions={{
                  color: isSelected ? '#38bdf8' : color,
                  weight: isSelected ? 8 : 5,
                  opacity: isSelected ? 1.0 : 0.85,
                  dashArray: road.data_coverage && road.data_coverage < 50 ? '6, 6' : undefined,
                }}
                eventHandlers={{
                  click: () => onRoadClick?.(road),
                }}
              >
                <Tooltip sticky>
                  <div className="font-sans text-xs">
                    <div className="font-bold text-gray-900">{road.name}</div>
                    <div className="text-gray-600">
                      Risk Score: <strong style={{ color }}>{Math.round(road.risk_score)}/100</strong>
                    </div>
                    <div className="text-[10px] text-gray-500">
                      Coverage: {road.data_coverage ? `${Math.round(road.data_coverage)}%` : 'Active'} • {road.status || 'Monitored'}
                    </div>
                  </div>
                </Tooltip>

                <Popup>
                  <div style={{ minWidth: 200, fontFamily: 'Inter, sans-serif' }}>
                    <div className="font-bold text-sm text-gray-900 border-b pb-1 mb-1">
                      {road.name}
                    </div>
                    <div className="text-xs text-gray-700 space-y-1">
                      <div>Calculated Risk: <strong style={{ color }}>{Math.round(road.risk_score)}/100</strong></div>
                      <div>SafeCity Score: <strong>{Math.round(road.safe_city_score)}/100</strong></div>
                      <div>Traffic Exposure: <strong>{road.traffic_exposure}</strong></div>
                      <div>Vulnerability: <strong>{road.vulnerability}</strong></div>
                      <div>Repair Status: <span className="font-semibold">{road.status || 'Normal'}</span></div>
                      {road.data_coverage && road.data_coverage < 50 && (
                        <div className="text-[10px] text-amber-700 bg-amber-50 p-1 rounded mt-1 border border-amber-200">
                          ⚠️ Low Coverage ({Math.round(road.data_coverage)}%). NO DATA ≠ SAFE ROAD.
                        </div>
                      )}
                    </div>
                    <button
                      onClick={() => onRoadClick?.(road)}
                      className="mt-2 w-full text-center py-1 bg-sky-600 text-white text-xs font-semibold rounded hover:bg-sky-700"
                    >
                      Inspect Road Dossier
                    </button>
                  </div>
                </Popup>
              </Polyline>
            )
          })}

        {/* 2. Hazards & Potholes Layer */}
        {showHazards &&
          hazards.map((hazard) => {
            const hColor = hazardColors[hazard.type] ?? '#3b82f6'
            const radius =
              hazard.severity === 'CRITICAL' ? 14 : hazard.severity === 'HIGH' ? 10 : 7

            return (
              <CircleMarker
                key={`hazard-${hazard.id}`}
                center={[hazard.latitude, hazard.longitude]}
                radius={radius}
                pathOptions={{
                  color: hColor,
                  fillColor: hColor,
                  fillOpacity: 0.8,
                  weight: 2,
                }}
                eventHandlers={{
                  click: () => onMarkerClick?.(hazard),
                }}
              >
                <Popup>
                  <div style={{ minWidth: 190, fontFamily: 'Inter, sans-serif' }}>
                    <div style={{ fontWeight: 700, fontSize: 13, marginBottom: 4 }}>
                      {hazardLabels[hazard.type] ?? hazard.type}
                    </div>
                    <div style={{ fontSize: 12, color: '#4b5563' }}>
                      Calculated Risk: <strong style={{ color: getRiskColor(hazard.risk_score) }}>{hazard.risk_score}/100</strong>
                    </div>
                    <div style={{ fontSize: 12, color: '#4b5563' }}>
                      Severity: <strong>{hazard.severity}</strong>
                    </div>
                    <div style={{ fontSize: 12, color: '#4b5563' }}>
                      Detection Confidence: {Math.round(hazard.confidence * 100)}%
                    </div>
                    {hazard.evidence_code && (
                      <div style={{ fontSize: 11, color: '#0284c7', fontFamily: 'monospace', marginTop: 2 }}>
                        Evidence: {hazard.evidence_code}
                      </div>
                    )}
                    <div style={{ fontSize: 11, color: '#9ca3af', marginTop: 4 }}>
                      Observed: {new Date(hazard.detected_at).toLocaleString()}
                    </div>
                    <div style={{ fontSize: 10, color: '#6b7280', marginTop: 3 }}>
                      ⚠️ SafeCity Local Edge Detection
                    </div>
                  </div>
                </Popup>
              </CircleMarker>
            )
          })}
      </MapContainer>
    </div>
  )
}
