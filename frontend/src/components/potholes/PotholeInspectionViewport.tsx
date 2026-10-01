import React, { useState, useRef, useEffect, useMemo } from 'react'
import type { PotholeAnalysisResult, PotholeDetectionItem } from '@/types'

interface PotholeInspectionViewportProps {
  potholeResult: PotholeAnalysisResult
  hoveredBoxIndex: number | null
  setHoveredBoxIndex: (idx: number | null) => void
  selectedBoxIndex: number | null
  setSelectedBoxIndex: (idx: number | null) => void
  viewMode: 'side-by-side' | 'single'
  setViewMode: (mode: 'side-by-side' | 'single') => void
  singleToggle: 'annotated' | 'original'
  setSingleToggle: (mode: 'annotated' | 'original') => void
}

interface ImagePanelProps {
  panelLabel: string
  subLabel: string
  imageUrl: string
  alt: string
  detections?: PotholeDetectionItem[]
  imageWidth?: number
  imageHeight?: number
  isAnnotated?: boolean
  renderMode?: 'vector' | 'raster'
  rasterUrl?: string
  onRenderModeChange?: (mode: 'vector' | 'raster') => void
  hoveredBoxIndex: number | null
  setHoveredBoxIndex: (idx: number | null) => void
  selectedBoxIndex: number | null
  setSelectedBoxIndex: (idx: number | null) => void
  isSingleView?: boolean
}

interface PlacedLabel {
  x: number // left in %
  y: number // top in %
  w: number // approx width in %
  h: number // approx height in %
}

// ── Collision Avoidance Engine ───────────────────────────────────────────────
function calculateCollisionFreeLabels(
  detections: PotholeDetectionItem[],
  imgW: number,
  imgH: number,
  renderedW: number,
  renderedH: number
): { x: number; y: number; placement: 'above' | 'inside-top' | 'below' | 'inside-bottom' | 'stagger-right' }[] {
  if (!detections || detections.length === 0) return []

  const placed: PlacedLabel[] = []
  const effectiveW = Math.max(renderedW, 100)
  const effectiveH = Math.max(renderedH, 100)

  // Approximate label dimensions in percentage (approx 68px wide by 18px tall)
  const labelWidthPct = Math.min(25, Math.max(6, (72 / effectiveW) * 100))
  const labelHeightPct = Math.min(8, Math.max(2.5, (19 / effectiveH) * 100))
  const paddingPct = 0.5 // small gap

  const overlaps = (r1: PlacedLabel, r2: PlacedLabel) => {
    return !(
      r1.x + r1.w <= r2.x ||
      r2.x + r2.w <= r1.x ||
      r1.y + r1.h <= r2.y ||
      r2.y + r2.h <= r1.y
    )
  }

  return detections.map((det) => {
    const { x1, y1, x2, y2 } = det.box
    const bx = Math.max(0, Math.min(100, (x1 / imgW) * 100))
    const by = Math.max(0, Math.min(100, (y1 / imgH) * 100))
    const bw = Math.max(0, Math.min(100 - bx, ((x2 - x1) / imgW) * 100))
    const bh = Math.max(0, Math.min(100 - by, ((y2 - y1) / imgH) * 100))

    // Candidate 1: Directly above the bounding box
    const candAbove: PlacedLabel = {
      x: Math.max(0, Math.min(100 - labelWidthPct, bx)),
      y: by - labelHeightPct - paddingPct,
      w: labelWidthPct,
      h: labelHeightPct,
    }

    // Candidate 2: Inside the top edge of the bounding box
    const candInsideTop: PlacedLabel = {
      x: Math.max(0, Math.min(100 - labelWidthPct, bx + paddingPct)),
      y: by + paddingPct,
      w: labelWidthPct,
      h: labelHeightPct,
    }

    // Candidate 3: Directly below the bounding box
    const candBelow: PlacedLabel = {
      x: Math.max(0, Math.min(100 - labelWidthPct, bx)),
      y: by + bh + paddingPct,
      w: labelWidthPct,
      h: labelHeightPct,
    }

    // Candidate 4: Inside the bottom edge of the bounding box
    const candInsideBottom: PlacedLabel = {
      x: Math.max(0, Math.min(100 - labelWidthPct, bx + paddingPct)),
      y: Math.max(0, by + bh - labelHeightPct - paddingPct),
      w: labelWidthPct,
      h: labelHeightPct,
    }

    // Candidate 5: Above the bounding box, shifted horizontally
    const candStaggerRight: PlacedLabel = {
      x: Math.max(0, Math.min(100 - labelWidthPct, bx + bw * 0.5)),
      y: by - labelHeightPct - paddingPct,
      w: labelWidthPct,
      h: labelHeightPct,
    }

    const testCandidate = (cand: PlacedLabel, minTop = 0, maxBottom = 100): boolean => {
      if (cand.y < minTop || cand.y + cand.h > maxBottom) return false
      for (const p of placed) {
        if (overlaps(cand, p)) return false
      }
      return true
    }

    let chosen = candAbove
    let placement: 'above' | 'inside-top' | 'below' | 'inside-bottom' | 'stagger-right' = 'above'

    if (testCandidate(candAbove, 0.5, 100)) {
      chosen = candAbove
      placement = 'above'
    } else if (testCandidate(candInsideTop, 0, 100) && candInsideTop.y + candInsideTop.h <= by + bh) {
      chosen = candInsideTop
      placement = 'inside-top'
    } else if (testCandidate(candBelow, 0, 99.5)) {
      chosen = candBelow
      placement = 'below'
    } else if (testCandidate(candInsideBottom, 0, 100)) {
      chosen = candInsideBottom
      placement = 'inside-bottom'
    } else if (testCandidate(candStaggerRight, 0.5, 100)) {
      chosen = candStaggerRight
      placement = 'stagger-right'
    } else {
      // Fallback: Clamp Candidate 1 within image bounds
      chosen = {
        x: Math.max(0, Math.min(100 - labelWidthPct, bx)),
        y: Math.max(0.5, Math.min(100 - labelHeightPct - 0.5, candAbove.y)),
        w: labelWidthPct,
        h: labelHeightPct,
      }
      placement = 'above'
    }

    placed.push(chosen)
    return { x: chosen.x, y: chosen.y, placement }
  })
}

// ── Single Image Panel (Aspect-Ratio Strict Containment) ────────────────────
function ImagePanel({
  panelLabel,
  subLabel,
  imageUrl,
  alt,
  detections = [],
  imageWidth,
  imageHeight,
  isAnnotated = false,
  renderMode = 'vector',
  rasterUrl,
  onRenderModeChange,
  hoveredBoxIndex,
  setHoveredBoxIndex,
  selectedBoxIndex,
  setSelectedBoxIndex,
  isSingleView = false,
}: ImagePanelProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [containerDims, setContainerDims] = useState<{ width: number; height: number }>({ width: 0, height: 0 })
  const [naturalDims, setNaturalDims] = useState<{ width: number; height: number }>({
    width: imageWidth || 0,
    height: imageHeight || 0,
  })

  // Measure container dimensions with ResizeObserver for exact pixel scaling
  useEffect(() => {
    const el = containerRef.current
    if (!el) return

    const updateDims = () => {
      const rect = el.getBoundingClientRect()
      // Subtract padding (16px total)
      const w = Math.max(10, rect.width - 16)
      const h = Math.max(10, rect.height - 16)
      setContainerDims({ width: w, height: h })
    }

    updateDims()
    const observer = new ResizeObserver(updateDims)
    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  const effectiveImgW = naturalDims.width || imageWidth || 1280
  const effectiveImgH = naturalDims.height || imageHeight || 720

  // Calculate strict aspect ratio preserving rendered size
  const { renderedW, renderedH } = useMemo(() => {
    const cW = containerDims.width || 600
    const cH = containerDims.height || 480
    if (cW <= 0 || cH <= 0 || effectiveImgW <= 0 || effectiveImgH <= 0) {
      return { renderedW: 600, renderedH: 400 }
    }
    const scale = Math.min(cW / effectiveImgW, cH / effectiveImgH)
    return {
      renderedW: Math.max(20, Math.floor(effectiveImgW * scale)),
      renderedH: Math.max(20, Math.floor(effectiveImgH * scale)),
    }
  }, [containerDims, effectiveImgW, effectiveImgH])

  // Compute collision-free label coordinates
  const labelPositions = useMemo(() => {
    if (!isAnnotated || renderMode === 'raster' || detections.length === 0) return []
    return calculateCollisionFreeLabels(detections, effectiveImgW, effectiveImgH, renderedW, renderedH)
  }, [isAnnotated, renderMode, detections, effectiveImgW, effectiveImgH, renderedW, renderedH])

  const displayUrl = renderMode === 'raster' && rasterUrl ? rasterUrl : imageUrl

  return (
    <div className="flex flex-col h-full bg-[#050b14] rounded-lg border border-[#14233c] overflow-hidden shadow-2xl">
      {/* Panel Subheader */}
      <div className="px-3.5 py-2 bg-[#091224] border-b border-[#14233c] flex items-center justify-between gap-2 text-xs select-none">
        <div className="flex items-center gap-2">
          <span className="font-mono font-bold tracking-wide text-gray-200 uppercase text-[11px] sm:text-xs">
            {panelLabel}
          </span>
          {isAnnotated && detections.length > 0 && (
            <span className="text-[10px] font-mono font-bold text-red-400 bg-red-950/70 border border-red-500/40 px-1.5 py-0.5 rounded">
              {detections.length} DETECTED
            </span>
          )}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-gray-400 bg-[#040810] px-2 py-0.5 rounded border border-[#1b2f4f]">
            {subLabel}
          </span>

          {/* Toggle between Vector High-Res Overlay and OpenCV Raster if annotated */}
          {isAnnotated && onRenderModeChange && (
            <div className="flex items-center bg-[#040810] p-0.5 rounded border border-[#1b2f4f]">
              <button
                type="button"
                onClick={() => onRenderModeChange('vector')}
                className={`text-[10px] px-2 py-0.5 rounded font-mono transition-colors ${
                  renderMode === 'vector'
                    ? 'bg-blue-600 text-white font-bold shadow-sm'
                    : 'text-gray-400 hover:text-gray-200'
                }`}
                title="Sub-pixel thin vector bounding boxes with collision-free labels"
              >
                Vector Overlay
              </button>
              <button
                type="button"
                onClick={() => onRenderModeChange('raster')}
                className={`text-[10px] px-2 py-0.5 rounded font-mono transition-colors ${
                  renderMode === 'raster'
                    ? 'bg-blue-600 text-white font-bold shadow-sm'
                    : 'text-gray-400 hover:text-gray-200'
                }`}
                title="Raw OpenCV raster burn-in"
              >
                OpenCV Raster
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Main Image Viewport Area: Full height, flex center, no distortion */}
      <div
        ref={containerRef}
        className={`w-full flex-1 min-h-[460px] sm:min-h-[500px] ${
          isSingleView ? 'h-[580px] lg:h-[700px] xl:h-[780px]' : 'h-[520px] lg:h-[620px] xl:h-[680px]'
        } bg-[#020711] flex items-center justify-center p-2 relative overflow-hidden select-none`}
      >
        {/* Aspect-Ratio Preserving Inner Wrapper: Strictly matches rendered image bounds */}
        <div
          className="relative flex items-center justify-center transition-all duration-75"
          style={{
            width: `${renderedW}px`,
            height: `${renderedH}px`,
            maxWidth: '100%',
            maxHeight: '100%',
          }}
        >
          {/* Base Clean / Rendered Image */}
          <img
            src={displayUrl}
            alt={alt}
            className="w-full h-full object-contain pointer-events-none block select-none"
            onLoad={(e) => {
              const nw = e.currentTarget.naturalWidth
              const nh = e.currentTarget.naturalHeight
              if (nw && nh && (nw !== naturalDims.width || nh !== naturalDims.height)) {
                setNaturalDims({ width: nw, height: nh })
              }
            }}
          />

          {/* Thin YOLO Bounding Boxes & Collision-Avoided Labels Overlay */}
          {isAnnotated && renderMode === 'vector' && detections.length > 0 && (
            <div className="absolute inset-0 pointer-events-none overflow-visible">
              {detections.map((det, idx) => {
                const { x1, y1, x2, y2 } = det.box
                const bx = Math.max(0, Math.min(100, (x1 / effectiveImgW) * 100))
                const by = Math.max(0, Math.min(100, (y1 / effectiveImgH) * 100))
                const bw = Math.max(0, Math.min(100 - bx, ((x2 - x1) / effectiveImgW) * 100))
                const bh = Math.max(0, Math.min(100 - by, ((y2 - y1) / effectiveImgH) * 100))

                const isHovered = hoveredBoxIndex === idx
                const isSelected = selectedBoxIndex === idx
                const labelPos = labelPositions[idx] || { x: bx, y: Math.max(0, by - 4) }

                return (
                  <React.Fragment key={`det-${idx}`}>
                    {/* Thin Bounding Box (1.5px border, subtle red glow, translucent background) */}
                    <div
                      className="absolute cursor-pointer transition-all duration-150 pointer-events-auto"
                      style={{
                        left: `${bx}%`,
                        top: `${by}%`,
                        width: `${bw}%`,
                        height: `${bh}%`,
                        border: isHovered || isSelected ? '1.5px solid #f87171' : '1.5px solid rgba(239, 68, 68, 0.95)',
                        backgroundColor: isHovered || isSelected ? 'rgba(239, 68, 68, 0.16)' : 'rgba(239, 68, 68, 0.07)',
                        boxShadow: isHovered || isSelected
                          ? '0 0 10px rgba(239, 68, 68, 0.8), inset 0 0 6px rgba(239, 68, 68, 0.25)'
                          : '0 0 5px rgba(239, 68, 68, 0.35)',
                        zIndex: isHovered || isSelected ? 30 : 10,
                      }}
                      onMouseEnter={() => setHoveredBoxIndex(idx)}
                      onMouseLeave={() => setHoveredBoxIndex(null)}
                      onClick={() => setSelectedBoxIndex(selectedBoxIndex === idx ? null : idx)}
                    >
                      {/* Subtle HUD corner accents (futuristic technical styling) */}
                      <div className="absolute -top-[1.5px] -left-[1.5px] w-1.5 h-1.5 border-t border-l border-red-300 pointer-events-none" />
                      <div className="absolute -top-[1.5px] -right-[1.5px] w-1.5 h-1.5 border-t border-r border-red-300 pointer-events-none" />
                      <div className="absolute -bottom-[1.5px] -left-[1.5px] w-1.5 h-1.5 border-b border-l border-red-300 pointer-events-none" />
                      <div className="absolute -bottom-[1.5px] -right-[1.5px] w-1.5 h-1.5 border-b border-r border-red-300 pointer-events-none" />
                    </div>

                    {/* Compact Detection Label (e.g. "Pothole 0.72", ~10px font, non-overlapping) */}
                    <div
                      className="absolute pointer-events-auto cursor-pointer select-none transition-all duration-150"
                      style={{
                        left: `${labelPos.x}%`,
                        top: `${labelPos.y}%`,
                        zIndex: isHovered || isSelected ? 40 : 20,
                      }}
                      onMouseEnter={() => setHoveredBoxIndex(idx)}
                      onMouseLeave={() => setHoveredBoxIndex(null)}
                      onClick={() => setSelectedBoxIndex(selectedBoxIndex === idx ? null : idx)}
                    >
                      <div
                        className={`flex items-center gap-1 font-mono font-semibold px-1.5 py-0.5 rounded text-[10px] leading-none tracking-tight transition-all duration-150 shadow-md ${
                          isHovered || isSelected
                            ? 'bg-red-950 text-white border border-red-400 shadow-red-500/40 ring-1 ring-red-400/50'
                            : 'bg-[#091224]/90 text-red-200 border border-red-500/50 backdrop-blur-sm'
                        }`}
                      >
                        <span className="w-1 h-1 rounded-full bg-red-400 inline-block animate-pulse" />
                        <span>Pothole {det.confidence.toFixed(2)}</span>
                      </div>
                    </div>
                  </React.Fragment>
                )
              })}
            </div>
          )}

          {/* Minimal HUD Metadata Badges (Bottom-Left and Bottom-Right of image) */}
          <div className="absolute bottom-1.5 left-1.5 pointer-events-none flex items-center gap-1.5">
            <span className="text-[9px] font-mono text-gray-400 bg-black/75 px-1.5 py-0.5 rounded border border-gray-800 backdrop-blur-sm">
              {effectiveImgW} × {effectiveImgH} px
            </span>
            <span className="text-[9px] font-mono text-gray-400 bg-black/75 px-1.5 py-0.5 rounded border border-gray-800 backdrop-blur-sm">
              ASPECT {(effectiveImgW / effectiveImgH).toFixed(2)}:1
            </span>
          </div>

          <div className="absolute bottom-1.5 right-1.5 pointer-events-none flex items-center gap-1.5">
            {isAnnotated ? (
              <span className="text-[9px] font-mono text-accent-light bg-black/75 px-1.5 py-0.5 rounded border border-blue-900/60 backdrop-blur-sm">
                YOLO MODEL ENGINE • CONF ≥ 0.25
              </span>
            ) : (
              <span className="text-[9px] font-mono text-gray-400 bg-black/75 px-1.5 py-0.5 rounded border border-gray-800 backdrop-blur-sm">
                RAW UNFILTERED SENSOR
              </span>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Main Inspection Viewport Component ──────────────────────────────────────
export default function PotholeInspectionViewport({
  potholeResult,
  hoveredBoxIndex,
  setHoveredBoxIndex,
  selectedBoxIndex,
  setSelectedBoxIndex,
  viewMode,
  setViewMode,
  singleToggle,
  setSingleToggle,
}: PotholeInspectionViewportProps) {
  const [renderMode, setRenderMode] = useState<'vector' | 'raster'>('vector')

  const detectionCount = potholeResult.detections ? potholeResult.detections.length : 0

  return (
    <div className="card p-0 overflow-hidden bg-[#030712] border border-[#14233c] shadow-2xl rounded-xl">
      {/* ── Viewport Top Bar ─────────────────────────────────────────────── */}
      <div className="px-4 py-3 bg-[#070e1b] border-b border-[#14233c] flex flex-wrap items-center justify-between gap-3 select-none">
        {/* Left: Viewport Title + Detection Count Badge */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse shadow-sm shadow-blue-500/50" />
            <span className="font-mono uppercase font-bold text-xs sm:text-sm tracking-wider text-gray-200">
              Visual Inspection Viewport
            </span>
          </div>

          {detectionCount > 0 ? (
            <span className="text-[11px] font-mono font-semibold text-blue-400 bg-blue-950/80 px-2.5 py-0.5 rounded border border-blue-800/60 shadow-sm">
              [{detectionCount} Detection Box(es)]
            </span>
          ) : (
            <span className="text-[11px] font-mono text-emerald-400 bg-emerald-950/60 px-2.5 py-0.5 rounded border border-emerald-800/50">
              [0 Detection Box(es) — Clear Surface]
            </span>
          )}
        </div>

        {/* Right: View Mode Switcher (Side-by-Side View vs Single View) */}
        <div className="flex items-center gap-1.5 bg-[#030712] p-1 rounded-lg border border-[#1b2f4f]">
          <button
            type="button"
            onClick={() => setViewMode('side-by-side')}
            className={`text-xs px-3 py-1.5 rounded-md font-mono transition-all duration-150 flex items-center gap-1.5 ${
              viewMode === 'side-by-side'
                ? 'bg-blue-600 text-white font-semibold shadow-md shadow-blue-600/30'
                : 'text-gray-400 hover:text-gray-200 hover:bg-navy-800/50'
            }`}
          >
            <span className="text-sm leading-none">◫</span>
            <span>Side-by-Side View</span>
          </button>
          <button
            type="button"
            onClick={() => setViewMode('single')}
            className={`text-xs px-3 py-1.5 rounded-md font-mono transition-all duration-150 flex items-center gap-1.5 ${
              viewMode === 'single'
                ? 'bg-blue-600 text-white font-semibold shadow-md shadow-blue-600/30'
                : 'text-gray-400 hover:text-gray-200 hover:bg-navy-800/50'
            }`}
          >
            <span className="text-sm leading-none">◻</span>
            <span>Single View</span>
          </button>
        </div>
      </div>

      {/* ── Viewport Content (Side-by-Side or Single View) ────────────────── */}
      {viewMode === 'side-by-side' ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 p-3 bg-[#030712]">
          {/* LEFT: Original Uploaded Image */}
          <ImagePanel
            panelLabel="LEFT: Original Uploaded Image"
            subLabel="Raw Capture"
            imageUrl={potholeResult.original_image_url}
            alt="Original Road Surface"
            imageWidth={potholeResult.image_width}
            imageHeight={potholeResult.image_height}
            isAnnotated={false}
            hoveredBoxIndex={hoveredBoxIndex}
            setHoveredBoxIndex={setHoveredBoxIndex}
            selectedBoxIndex={selectedBoxIndex}
            setSelectedBoxIndex={setSelectedBoxIndex}
            isSingleView={false}
          />

          {/* RIGHT: AI Annotated Analysis */}
          <ImagePanel
            panelLabel="RIGHT: AI Annotated Analysis"
            subLabel="YOLO BBoxes + Labels"
            imageUrl={potholeResult.original_image_url}
            rasterUrl={potholeResult.processed_image_url}
            alt="AI Annotated Road Surface"
            detections={potholeResult.detections}
            imageWidth={potholeResult.image_width}
            imageHeight={potholeResult.image_height}
            isAnnotated={true}
            renderMode={renderMode}
            onRenderModeChange={setRenderMode}
            hoveredBoxIndex={hoveredBoxIndex}
            setHoveredBoxIndex={setHoveredBoxIndex}
            selectedBoxIndex={selectedBoxIndex}
            setSelectedBoxIndex={setSelectedBoxIndex}
            isSingleView={false}
          />
        </div>
      ) : (
        /* Single View Mode */
        <div className="p-3 bg-[#030712] flex flex-col gap-3">
          {/* Single View Layer Switcher */}
          <div className="flex items-center justify-between bg-[#070e1b] px-3.5 py-2 rounded-lg border border-[#14233c]">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-gray-400">ACTIVE LAYER:</span>
              <div className="flex items-center bg-[#030712] p-0.5 rounded border border-[#1b2f4f]">
                <button
                  type="button"
                  onClick={() => setSingleToggle('annotated')}
                  className={`text-xs px-3 py-1 rounded font-mono transition-colors ${
                    singleToggle === 'annotated'
                      ? 'bg-blue-600 text-white font-bold shadow-sm'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  AI Annotated Analysis
                </button>
                <button
                  type="button"
                  onClick={() => setSingleToggle('original')}
                  className={`text-xs px-3 py-1 rounded font-mono transition-colors ${
                    singleToggle === 'original'
                      ? 'bg-blue-600 text-white font-bold shadow-sm'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  Original Photo
                </button>
              </div>
            </div>

            <span className="text-[11px] font-mono text-gray-400">
              {singleToggle === 'annotated' ? 'Full Bounding Box Inspection' : 'Raw Sensor Frame'}
            </span>
          </div>

          <ImagePanel
            panelLabel={
              singleToggle === 'annotated'
                ? 'AI Annotated Visual Inspection'
                : 'Original Raw Frame'
            }
            subLabel={singleToggle === 'annotated' ? 'YOLO BBoxes + Labels' : 'Raw Capture'}
            imageUrl={potholeResult.original_image_url}
            rasterUrl={potholeResult.processed_image_url}
            alt="Road Surface Inspection"
            detections={singleToggle === 'annotated' ? potholeResult.detections : []}
            imageWidth={potholeResult.image_width}
            imageHeight={potholeResult.image_height}
            isAnnotated={singleToggle === 'annotated'}
            renderMode={renderMode}
            onRenderModeChange={setRenderMode}
            hoveredBoxIndex={hoveredBoxIndex}
            setHoveredBoxIndex={setHoveredBoxIndex}
            selectedBoxIndex={selectedBoxIndex}
            setSelectedBoxIndex={setSelectedBoxIndex}
            isSingleView={true}
          />
        </div>
      )}
    </div>
  )
}
