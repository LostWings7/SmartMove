import React, { useState } from 'react'
import type { RoadConditionAnalysisResult, RoadConditionObservation } from '@/types'
import RiskBadge from '@/components/ui/RiskBadge'
import StatusBadge from '@/components/ui/StatusBadge'

interface RoadConditionInspectionViewportProps {
  result: RoadConditionAnalysisResult
  selectedObservationIndex: number | null
  setSelectedObservationIndex: (idx: number | null) => void
}

export default function RoadConditionInspectionViewport({
  result,
  selectedObservationIndex,
  setSelectedObservationIndex,
}: RoadConditionInspectionViewportProps) {
  const [activeLayer, setActiveLayer] = useState<'fused' | 'potholes' | 'water' | 'cracks' | 'original'>('fused')
  const [viewMode, setViewMode] = useState<'side-by-side' | 'single'>('side-by-side')

  // Resolve layer URL
  const getActiveImageUrl = () => {
    if (activeLayer === 'original') return result.original_image_url
    if (activeLayer === 'fused') return result.layer_urls.fused || result.processed_image_url
    if (activeLayer === 'potholes') return result.layer_urls.potholes || result.processed_image_url
    if (activeLayer === 'water') return result.layer_urls.water || result.processed_image_url
    if (activeLayer === 'cracks') return result.layer_urls.cracks || result.processed_image_url
    return result.processed_image_url
  }

  const selectedObs: RoadConditionObservation | null =
    selectedObservationIndex !== null && result.observations[selectedObservationIndex]
      ? result.observations[selectedObservationIndex]
      : null

  return (
    <div className="space-y-4">
      {/* View Mode & Multi-Layer Control Header */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-navy-800/80 rounded-xl border border-gray-800">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-gray-400">ACTIVE LAYER:</span>
          <div className="flex items-center bg-navy-950 p-1 rounded-lg border border-gray-800 gap-1">
            <button
              onClick={() => setActiveLayer('fused')}
              className={`px-3 py-1 text-xs font-bold rounded transition-all ${
                activeLayer === 'fused'
                  ? 'bg-accent text-white shadow-sm shadow-accent/40'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              ⬟ Fused AI ({result.total_hazards_count})
            </button>
            <button
              onClick={() => setActiveLayer('potholes')}
              className={`px-2.5 py-1 text-xs font-semibold rounded transition-all ${
                activeLayer === 'potholes'
                  ? 'bg-red-600/80 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Potholes ({result.pothole_count})
            </button>
            <button
              onClick={() => setActiveLayer('water')}
              className={`px-2.5 py-1 text-xs font-semibold rounded transition-all ${
                activeLayer === 'water'
                  ? 'bg-cyan-600/80 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Water Mask ({result.water_regions_count})
            </button>
            <button
              onClick={() => setActiveLayer('cracks')}
              className={`px-2.5 py-1 text-xs font-semibold rounded transition-all ${
                activeLayer === 'cracks'
                  ? 'bg-amber-600/80 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Cracks ({result.crack_count})
            </button>
            <button
              onClick={() => setActiveLayer('original')}
              className={`px-2 py-1 text-xs font-semibold rounded transition-all ${
                activeLayer === 'original'
                  ? 'bg-gray-700 text-white'
                  : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              Original
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400 font-semibold">VIEWPORT:</span>
          <div className="flex items-center bg-navy-950 p-1 rounded-lg border border-gray-800 text-xs">
            <button
              onClick={() => setViewMode('side-by-side')}
              className={`px-2.5 py-1 rounded font-semibold transition-all ${
                viewMode === 'side-by-side' ? 'bg-navy-700 text-white' : 'text-gray-400 hover:text-white'
              }`}
            >
              Side-by-Side
            </button>
            <button
              onClick={() => setViewMode('single')}
              className={`px-2.5 py-1 rounded font-semibold transition-all ${
                viewMode === 'single' ? 'bg-navy-700 text-white' : 'text-gray-400 hover:text-white'
              }`}
            >
              Single
            </button>
          </div>
        </div>
      </div>

      {/* Visual Canvas Viewport */}
      <div
        className={`grid gap-4 ${
          viewMode === 'side-by-side' ? 'grid-cols-1 lg:grid-cols-2' : 'grid-cols-1'
        }`}
      >
        {/* Left Panel: Original Image (in side-by-side) */}
        {viewMode === 'side-by-side' && (
          <div className="card p-3 border-gray-800 flex flex-col">
            <div className="flex items-center justify-between pb-2 mb-2 border-b border-gray-800 text-xs">
              <span className="font-semibold text-gray-300">ORIGINAL ROAD CAPTURE</span>
              <span className="font-mono text-gray-500 text-[11px]">{result.image_id}</span>
            </div>
            <div className="relative rounded-lg overflow-hidden bg-navy-950 aspect-video flex items-center justify-center border border-gray-800/80">
              <img
                src={result.original_image_url}
                alt="Original road corridor capture"
                className="w-full h-full object-contain"
              />
            </div>
          </div>
        )}

        {/* Right Panel (or Single): Active Multi-Layer Annotation */}
        <div className="card p-3 border-gray-800 flex flex-col">
          <div className="flex items-center justify-between pb-2 mb-2 border-b border-gray-800 text-xs">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-white uppercase tracking-wider">
                {activeLayer === 'fused' && 'Multimodal Fused Inspection Layer'}
                {activeLayer === 'potholes' && 'Pothole Subsystem Layer'}
                {activeLayer === 'water' && 'Waterlogging & Ponding Segmentation'}
                {activeLayer === 'cracks' && 'Road Crack Detection Layer'}
                {activeLayer === 'original' && 'Raw Visual Layer'}
              </span>
              <span className="text-[10px] bg-accent/20 text-accent-light px-1.5 py-0.5 rounded font-mono">
                {result.processing_time_ms} ms
              </span>
            </div>
            <span className="text-gray-500 font-mono text-[11px]">{result.evidence_id}</span>
          </div>

          <div className="relative rounded-lg overflow-hidden bg-navy-950 aspect-video flex items-center justify-center border border-gray-800/80">
            <img
              src={getActiveImageUrl()}
              alt="Active Road Condition Layer"
              className="w-full h-full object-contain"
            />

            {/* Quick overlay badges on canvas */}
            <div className="absolute top-2 left-2 flex flex-col gap-1.5 pointer-events-none">
              {result.water_filled_pothole_count > 0 && (
                <div className="bg-cyan-500/90 text-navy-950 font-black text-[10px] px-2 py-0.5 rounded shadow-lg backdrop-blur-sm border border-cyan-300">
                  ⚡ {result.water_filled_pothole_count} WATER-FILLED POTHOLE(S) DETECTED
                </div>
              )}
              {result.water_coverage_percent > 0 && (
                <div className="bg-navy-900/90 text-cyan-300 text-[10px] px-2 py-0.5 rounded border border-cyan-500/40 backdrop-blur-sm font-mono">
                  Water Coverage: {result.water_coverage_percent.toFixed(1)}% of visible road ({result.water_severity})
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Defect Category KPI Cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-2">
        <div className="card p-3 border-gray-800">
          <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Potholes</div>
          <div className="text-xl font-black text-red-400 mt-0.5">{result.pothole_count}</div>
          <div className="text-[10px] text-gray-400 mt-0.5">Protected Subsystem</div>
        </div>

        <div className="card p-3 border-gray-800">
          <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Water Areas</div>
          <div className="text-xl font-black text-cyan-400 mt-0.5">{result.water_regions_count}</div>
          <div className="text-[10px] text-gray-400 mt-0.5">{result.water_coverage_percent.toFixed(1)}% visible road</div>
        </div>

        <div className="card p-3 border-cyan-500/30 bg-cyan-950/20">
          <div className="text-[10px] font-semibold text-cyan-300 uppercase tracking-wider">Water-Filled</div>
          <div className="text-xl font-black text-cyan-300 mt-0.5">{result.water_filled_pothole_count}</div>
          <div className="text-[10px] text-cyan-400/80 mt-0.5">Overlap ≥ 20% Cutoff</div>
        </div>

        <div className="card p-3 border-gray-800">
          <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Longitudinal</div>
          <div className="text-xl font-black text-amber-400 mt-0.5">{result.longitudinal_crack_count}</div>
          <div className="text-[10px] text-gray-400 mt-0.5">Parallel to traffic</div>
        </div>

        <div className="card p-3 border-gray-800">
          <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Transverse</div>
          <div className="text-xl font-black text-amber-400 mt-0.5">{result.transverse_crack_count}</div>
          <div className="text-[10px] text-gray-400 mt-0.5">Perpendicular cracks</div>
        </div>

        <div className="card p-3 border-gray-800">
          <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wider">Alligator Cracks</div>
          <div className="text-xl font-black text-orange-400 mt-0.5">{result.alligator_crack_count}</div>
          <div className="text-[10px] text-gray-400 mt-0.5">Fatigue pattern</div>
        </div>
      </div>

      {/* Observations Dossier & Explainability Breakdown */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left: Observations List */}
        <div className="lg:col-span-2 card p-4 border-gray-800 space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-gray-800">
            <div>
              <h3 className="text-xs font-bold text-white uppercase tracking-wider">
                Road Condition Defect Dossier ({result.observations.length})
              </h3>
              <p className="text-[11px] text-gray-400">
                Click any observation to inspect evidence, fusion formula, and confidence.
              </p>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-gray-400">Fused Risk:</span>
              <span className="text-sm font-black text-white font-mono">{result.overall_risk_score}/100</span>
              <RiskBadge level={result.highest_severity as any} />
            </div>
          </div>

          <div className="space-y-2 max-h-[380px] overflow-y-auto pr-1">
            {result.observations.length === 0 ? (
              <div className="p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs">
                ✓ Road corridor clear of verified defects. Zero potholes, standing water, or severe structural cracks detected.
              </div>
            ) : (
              result.observations.map((obs, idx) => {
                const isSelected = selectedObservationIndex === idx
                const isWaterFilled = obs.hazard_type === 'water_filled_pothole'
                const isPothole = obs.hazard_type === 'pothole'
                const isWater = obs.hazard_type === 'water'

                return (
                  <div
                    key={obs.observation_id || idx}
                    onClick={() => setSelectedObservationIndex(isSelected ? null : idx)}
                    className={`p-3 rounded-lg border text-xs cursor-pointer transition-all ${
                      isSelected
                        ? 'border-accent bg-accent/10 shadow-sm'
                        : isWaterFilled
                        ? 'border-cyan-500/40 bg-cyan-950/20 hover:border-cyan-500'
                        : 'border-gray-800 bg-navy-900/60 hover:border-gray-700'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-[10px] text-gray-500">#{idx + 1}</span>
                        <span className="font-bold text-white capitalize">
                          {obs.hazard_type.replace(/_/g, ' ')}
                        </span>
                        {isWaterFilled && (
                          <span className="text-[10px] bg-cyan-500/20 text-cyan-300 font-semibold px-2 py-0.5 rounded border border-cyan-500/30">
                            FUSION: Pothole ∩ Water
                          </span>
                        )}
                        {isWater && obs.coverage_percent && (
                          <span className="text-[10px] bg-blue-500/20 text-blue-300 font-mono px-1.5 py-0.5 rounded">
                            {obs.coverage_percent}% coverage
                          </span>
                        )}
                      </div>

                      <div className="flex items-center gap-2">
                        <span className="text-[10px] font-mono text-gray-400">
                          Conf: {(obs.confidence * 100).toFixed(0)}%
                        </span>
                        <RiskBadge level={obs.severity as any} />
                      </div>
                    </div>

                    {isWaterFilled && obs.fusion_evidence && (
                      <div className="mt-2 text-[11px] p-2 rounded bg-navy-950 border border-cyan-500/20 text-cyan-200/90 font-mono">
                        {obs.fusion_evidence.fusion_formula} • Overlap: {obs.fusion_evidence.overlap_percent}%
                      </div>
                    )}
                  </div>
                )
              })
            )}
          </div>
        </div>

        {/* Right: Selected Defect Inspector & Verification Rules */}
        <div className="card p-4 border-gray-800 space-y-3">
          <div className="card-header pb-2 border-b border-gray-800 text-xs">
            Observation Verification & Rules
          </div>

          {selectedObs ? (
            <div className="space-y-2.5 text-xs">
              <div className="flex justify-between py-1 border-b border-gray-800/60">
                <span className="text-gray-400">Observation ID</span>
                <span className="text-white font-mono">{selectedObs.observation_id}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-gray-800/60">
                <span className="text-gray-400">Hazard Type</span>
                <span className="text-accent font-semibold capitalize">{selectedObs.hazard_type.replace(/_/g, ' ')}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-gray-800/60">
                <span className="text-gray-400">Source Model</span>
                <span className="text-white font-mono text-[11px]">{selectedObs.source_model} (v{selectedObs.model_version})</span>
              </div>
              <div className="flex justify-between py-1 border-b border-gray-800/60">
                <span className="text-gray-400">Raw Model Confidence</span>
                <span className="text-emerald-400 font-mono font-bold">
                  {((selectedObs.raw_model_confidence ?? selectedObs.confidence) * 100).toFixed(1)}%
                </span>
              </div>
              {selectedObs.road_overlap_ratio !== undefined && (
                <div className="flex justify-between py-1 border-b border-gray-800/60">
                  <span className="text-gray-400">Road Overlap Ratio</span>
                  <span className="text-cyan-300 font-mono">{(selectedObs.road_overlap_ratio * 100).toFixed(1)}%</span>
                </div>
              )}
              {selectedObs.road_context_score !== undefined && (
                <div className="flex justify-between py-1 border-b border-gray-800/60">
                  <span className="text-gray-400">Road Context Score</span>
                  <span className="text-amber-300 font-mono">{(selectedObs.road_context_score * 100).toFixed(1)}%</span>
                </div>
              )}
              {selectedObs.final_acceptance_decision && (
                <div className="flex justify-between py-1 border-b border-gray-800/60">
                  <span className="text-gray-400">Acceptance Rule</span>
                  <span className="text-gray-200 font-mono text-[10px]">{selectedObs.final_acceptance_decision}</span>
                </div>
              )}
              <div className="flex justify-between py-1 border-b border-gray-800/60">
                <span className="text-gray-400">Hazard Severity</span>
                <RiskBadge level={selectedObs.severity as any} />
              </div>
              <div className="flex justify-between py-1 border-b border-gray-800/60">
                <span className="text-gray-400">Individual Risk Contribution</span>
                <span className="text-white font-mono font-bold">{selectedObs.risk_score} / 100</span>
              </div>

              {selectedObs.fusion_evidence && (
                <div className="p-2.5 rounded bg-cyan-950/40 border border-cyan-500/30 text-[11px] space-y-1">
                  <div className="font-semibold text-cyan-300">Evidence Fusion Audit:</div>
                  <div className="text-gray-300">Pothole Confidence: {(selectedObs.fusion_evidence.pothole_confidence * 100).toFixed(1)}%</div>
                  <div className="text-gray-300">Water Confidence: {(selectedObs.fusion_evidence.water_confidence * 100).toFixed(1)}%</div>
                  <div className="text-gray-300">Overlap Ratio: {selectedObs.fusion_evidence.overlap_percent}%</div>
                  <div className="text-cyan-400 font-mono text-[10px] mt-1">{selectedObs.fusion_evidence.fusion_formula}</div>
                </div>
              )}
            </div>
          ) : (
            <div className="text-xs text-gray-500 space-y-3">
              <p>Select any item in the defect dossier to inspect individual model confidence, bounding box coordinates, and evidence fusion audit trail.</p>
              <div className="p-3 bg-navy-900 rounded border border-gray-800/60 space-y-1.5 text-[11px]">
                <div className="font-semibold text-gray-300">Explainable Fusion Formula:</div>
                <p className="text-gray-400">
                  Water-filled potholes are confirmed only when the intersection of the pothole bounding area and the water segmentation mask exceeds 20%.
                </p>
                <div className="font-semibold text-gray-300 pt-1">SafeCity Risk Engine:</div>
                <p className="text-gray-400">
                  Individual hazard risks are fused using non-linear diminishing marginal risk to strictly avoid double-counting.
                </p>
              </div>
            </div>
          )}

          <div className="pt-2 border-t border-gray-800/60 text-[10px] text-gray-500">
            {result.disclaimer}
          </div>
        </div>
      </div>
    </div>
  )
}
