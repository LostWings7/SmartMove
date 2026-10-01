import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import Layout from '@/components/layout/Layout'
import LocationPickerMap from '@/components/complaints/LocationPickerMap'
import { useCategories, useSubmitComplaint, useUploadEvidence } from '@/hooks/useComplaintsApi'
import type { ComplaintCategory } from '@/types'

export default function CreateComplaint() {
  const navigate = useNavigate()
  const { data: categories = [], isLoading: loadingCategories } = useCategories()
  const submitMutation = useSubmitComplaint()
  const uploadEvidenceMutation = useUploadEvidence()

  const [step, setStep] = useState<1 | 2 | 3 | 4>(1)

  // Form state
  const [selectedCategoryId, setSelectedCategoryId] = useState<number | null>(null)
  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [priority, setPriority] = useState('HIGH')

  // Location state (defaults to School Road coordinates)
  const [latitude, setLatitude] = useState(51.5074)
  const [longitude, setLongitude] = useState(-0.1278)
  const [locationAddress, setLocationAddress] = useState('School Road Corridor, East Crosswalk')
  const [gpsLoading, setGpsLoading] = useState(false)

  // Evidence files
  const [files, setFiles] = useState<File[]>([])
  const [filePreviews, setFilePreviews] = useState<string[]>([])

  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const selectedCategory = categories.find((c) => c.id === selectedCategoryId)

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const selected = e.target.files
    if (!selected) return

    const newFiles = Array.from(selected)
    const combined = [...files, ...newFiles].slice(0, 5) // up to 5 files
    setFiles(combined)

    // Generate previews
    const previews = combined.map((f) => URL.createObjectURL(f))
    setFilePreviews(previews)
  }

  function removeFile(index: number) {
    const updatedFiles = files.filter((_, i) => i !== index)
    const updatedPreviews = filePreviews.filter((_, i) => i !== index)
    setFiles(updatedFiles)
    setFilePreviews(updatedPreviews)
  }

  function handleUseMyLocation() {
    if (!navigator.geolocation) {
      alert('Geolocation is not supported by your browser.')
      return
    }
    setGpsLoading(true)
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const lat = Number(pos.coords.latitude.toFixed(6))
        const lng = Number(pos.coords.longitude.toFixed(6))
        setLatitude(lat)
        setLongitude(lng)
        setLocationAddress(`Current Device Location (${lat}, ${lng})`)
        setGpsLoading(false)
      },
      (err) => {
        console.warn('Geolocation failed:', err)
        setGpsLoading(false)
        alert('Could not acquire GPS position. You can click on the map to set location.')
      },
      { timeout: 10000, enableHighAccuracy: true }
    )
  }

  async function handleFinalSubmit() {
    if (!selectedCategoryId || !title.trim() || !description.trim()) {
      setError('Please complete all required fields.')
      return
    }

    setError(null)
    setIsSubmitting(true)
    try {
      const complaint = await submitMutation.mutateAsync({
        category_id: selectedCategoryId,
        title: title.trim(),
        description: description.trim(),
        location_lat: latitude,
        location_lng: longitude,
        location_address: locationAddress.trim(),
        priority,
      })

      // Upload evidence files if any
      if (files.length > 0) {
        for (const file of files) {
          try {
            await uploadEvidenceMutation.mutateAsync({
              complaintId: complaint.id,
              file,
              notes: 'Uploaded during initial citizen report',
            })
          } catch (uploadErr) {
            console.error('Evidence upload error:', uploadErr)
          }
        }
      }

      navigate(`/citizen/complaints/${complaint.id}`, {
        state: { newSubmitted: true, publicReference: complaint.public_reference },
      })
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to submit complaint. Please try again.')
      setIsSubmitting(false)
    }
  }

  return (
    <Layout
      title="File a Citizen Complaint"
      subtitle="Report road cavities, waterlogging, or hazards directly to municipal engineering"
    >
      <div className="max-w-3xl mx-auto space-y-6">
        {/* Step Indicator Header */}
        <div className="bg-navy-800 border border-gray-800 rounded-2xl p-4 sm:p-5">
          <div className="flex items-center justify-between">
            {[
              { num: 1, label: 'Problem' },
              { num: 2, label: 'Location' },
              { num: 3, label: 'Evidence' },
              { num: 4, label: 'Review & Submit' },
            ].map((s) => (
              <div
                key={s.num}
                className={`flex items-center gap-2 ${
                  step === s.num
                    ? 'text-accent font-semibold'
                    : step > s.num
                    ? 'text-emerald-400 font-medium'
                    : 'text-gray-500'
                }`}
              >
                <div
                  className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-mono font-bold ${
                    step === s.num
                      ? 'bg-accent text-white shadow-md shadow-accent/30'
                      : step > s.num
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
                      : 'bg-navy-900 border border-gray-700 text-gray-500'
                  }`}
                >
                  {step > s.num ? '✓' : s.num}
                </div>
                <span className="text-xs hidden sm:inline">{s.label}</span>
              </div>
            ))}
          </div>
        </div>

        {error && (
          <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-xs flex items-center gap-2">
            <span>⚠</span>
            <span>{error}</span>
          </div>
        )}

        {/* Step 1: Problem Definition */}
        {step === 1 && (
          <div className="bg-navy-800 border border-gray-800 rounded-2xl p-6 sm:p-8 space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-white">Step 1: What is the issue?</h3>
              <p className="text-gray-400 text-xs mt-0.5">Select the category that best describes the road safety hazard.</p>
            </div>

            {loadingCategories ? (
              <div className="text-gray-500 text-xs">Loading categories...</div>
            ) : (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {categories.map((cat) => {
                  const isSelected = selectedCategoryId === cat.id
                  return (
                    <button
                      key={cat.id}
                      type="button"
                      onClick={() => setSelectedCategoryId(cat.id)}
                      className={`text-left p-3.5 rounded-xl border transition ${
                        isSelected
                          ? 'bg-accent/15 border-accent text-white shadow-md shadow-accent/10'
                          : 'bg-navy-900/60 border-gray-800 hover:border-gray-700 text-gray-300'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="text-sm font-semibold">{cat.name}</div>
                        {cat.is_ai_supported && (
                          <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                            AI-Assisted
                          </span>
                        )}
                      </div>
                      {cat.description && (
                        <p className="text-[11px] text-gray-400 mt-1 line-clamp-2">{cat.description}</p>
                      )}
                    </button>
                  )
                })}
              </div>
            )}

            <div className="space-y-4 pt-2">
              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1.5">
                  Complaint Title <span className="text-red-400">*</span>
                </label>
                <input
                  type="text"
                  required
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="e.g. Deep pothole right before zebra crossing"
                  className="w-full bg-navy-900 border border-gray-700 rounded-xl px-3.5 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1.5">
                  Detailed Description <span className="text-red-400">*</span>
                </label>
                <textarea
                  rows={4}
                  required
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Describe the defect, visible dimensions, impact on traffic, or danger to pedestrians..."
                  className="w-full bg-navy-900 border border-gray-700 rounded-xl px-3.5 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1.5">
                  Perceived Problem Severity
                </label>
                <div className="grid grid-cols-4 gap-2">
                  {['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'].map((lvl) => (
                    <button
                      key={lvl}
                      type="button"
                      onClick={() => setPriority(lvl)}
                      className={`py-2 rounded-xl text-xs font-semibold border transition ${
                        priority === lvl
                          ? lvl === 'CRITICAL'
                            ? 'bg-rose-500/20 text-rose-400 border-rose-500'
                            : lvl === 'HIGH'
                            ? 'bg-amber-500/20 text-amber-400 border-amber-500'
                            : 'bg-accent/20 text-accent border-accent'
                          : 'bg-navy-900 border-gray-800 text-gray-400 hover:border-gray-700'
                      }`}
                    >
                      {lvl}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="flex justify-end pt-4 border-t border-gray-800">
              <button
                type="button"
                disabled={!selectedCategoryId || !title.trim() || !description.trim()}
                onClick={() => setStep(2)}
                className="px-6 py-2.5 bg-accent hover:bg-accent/90 disabled:opacity-50 text-white text-xs font-semibold rounded-xl transition shadow-lg shadow-accent/20"
              >
                Next: Select Location →
              </button>
            </div>
          </div>
        )}

        {/* Step 2: Location Map */}
        {step === 2 && (
          <div className="bg-navy-800 border border-gray-800 rounded-2xl p-6 sm:p-8 space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h3 className="text-lg font-semibold text-white">Step 2: Where is the defect?</h3>
                <p className="text-gray-400 text-xs mt-0.5">Click the map to place the marker at the exact complaint position.</p>
              </div>

              <button
                type="button"
                onClick={handleUseMyLocation}
                disabled={gpsLoading}
                className="inline-flex items-center gap-2 px-3.5 py-2 bg-navy-700 hover:bg-navy-600 border border-gray-700 text-xs text-white rounded-xl transition"
              >
                <span>📍</span>
                <span>{gpsLoading ? 'Acquiring GPS...' : 'Use Current Location'}</span>
              </button>
            </div>

            {/* Interactive Leaflet Map */}
            <LocationPickerMap
              latitude={latitude}
              longitude={longitude}
              onChangeLocation={(lat, lng) => {
                setLatitude(lat)
                setLongitude(lng)
              }}
            />

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-gray-400 mb-1">Coordinates</label>
                <div className="bg-navy-900 border border-gray-700 rounded-xl px-3.5 py-2.5 text-xs font-mono text-cyan-400">
                  {latitude.toFixed(6)}, {longitude.toFixed(6)}
                </div>
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-400 mb-1">Address / Landmark</label>
                <input
                  type="text"
                  value={locationAddress}
                  onChange={(e) => setLocationAddress(e.target.value)}
                  placeholder="e.g. Near Market Crossroad bus shelter"
                  className="w-full bg-navy-900 border border-gray-700 rounded-xl px-3.5 py-2.5 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-accent"
                />
              </div>
            </div>

            <div className="flex items-center justify-between pt-4 border-t border-gray-800">
              <button
                type="button"
                onClick={() => setStep(1)}
                className="px-5 py-2 text-xs text-gray-400 hover:text-white"
              >
                ← Back
              </button>
              <button
                type="button"
                onClick={() => setStep(3)}
                className="px-6 py-2.5 bg-accent hover:bg-accent/90 text-white text-xs font-semibold rounded-xl transition shadow-lg shadow-accent/20"
              >
                Next: Add Evidence →
              </button>
            </div>
          </div>
        )}

        {/* Step 3: Evidence Upload */}
        {step === 3 && (
          <div className="bg-navy-800 border border-gray-800 rounded-2xl p-6 sm:p-8 space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-white">Step 3: Attach Evidence</h3>
              <p className="text-gray-400 text-xs mt-0.5">
                Upload clear photographs or video clips of the defect (up to 5 files, 15MB each).
              </p>
            </div>

            {/* Dropzone */}
            <label className="block border-2 border-dashed border-gray-700 hover:border-accent/60 bg-navy-900/60 rounded-2xl p-8 text-center cursor-pointer transition">
              <div className="text-4xl mb-2">📸</div>
              <div className="text-sm font-medium text-white">Click or drag photos/videos here</div>
              <div className="text-xs text-gray-400 mt-1">Supported formats: JPG, PNG, WEBP, MP4</div>
              <input
                type="file"
                multiple
                accept="image/*,video/*"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>

            {/* Previews */}
            {filePreviews.length > 0 && (
              <div className="space-y-2">
                <div className="text-xs font-medium text-gray-300">Selected Files ({files.length}/5)</div>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                  {files.map((file, idx) => (
                    <div
                      key={idx}
                      className="relative bg-navy-900 border border-gray-700 rounded-xl overflow-hidden group p-2 flex flex-col items-center"
                    >
                      {file.type.startsWith('image/') ? (
                        <img
                          src={filePreviews[idx]}
                          alt={file.name}
                          className="w-full h-24 object-cover rounded-lg mb-2"
                        />
                      ) : (
                        <div className="w-full h-24 bg-navy-950 flex items-center justify-center text-2xl rounded-lg mb-2">
                          🎥
                        </div>
                      )}
                      <div className="w-full text-[11px] text-gray-300 truncate text-center">{file.name}</div>
                      <div className="text-[10px] text-gray-500 font-mono">
                        {(file.size / (1024 * 1024)).toFixed(2)} MB
                      </div>
                      <button
                        type="button"
                        onClick={() => removeFile(idx)}
                        className="absolute top-3 right-3 bg-black/70 hover:bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs transition"
                        title="Remove file"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="flex items-center justify-between pt-4 border-t border-gray-800">
              <button
                type="button"
                onClick={() => setStep(2)}
                className="px-5 py-2 text-xs text-gray-400 hover:text-white"
              >
                ← Back
              </button>
              <button
                type="button"
                onClick={() => setStep(4)}
                className="px-6 py-2.5 bg-accent hover:bg-accent/90 text-white text-xs font-semibold rounded-xl transition shadow-lg shadow-accent/20"
              >
                Next: Review & Submit →
              </button>
            </div>
          </div>
        )}

        {/* Step 4: Review & Final Submission */}
        {step === 4 && (
          <div className="bg-navy-800 border border-gray-800 rounded-2xl p-6 sm:p-8 space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-white">Step 4: Review & Submit</h3>
              <p className="text-gray-400 text-xs mt-0.5">Please review your complaint summary before authoritative database filing.</p>
            </div>

            <div className="bg-navy-900 border border-gray-800 rounded-xl p-5 space-y-4 text-xs">
              <div className="flex items-center justify-between border-b border-gray-800 pb-3">
                <span className="text-gray-400">Category</span>
                <span className="font-semibold text-white">{selectedCategory?.name}</span>
              </div>

              <div className="border-b border-gray-800 pb-3 space-y-1">
                <span className="text-gray-400 block">Title</span>
                <span className="font-semibold text-white text-sm block">{title}</span>
              </div>

              <div className="border-b border-gray-800 pb-3 space-y-1">
                <span className="text-gray-400 block">Description</span>
                <p className="text-gray-300 leading-relaxed">{description}</p>
              </div>

              <div className="grid grid-cols-2 gap-4 border-b border-gray-800 pb-3">
                <div>
                  <span className="text-gray-400 block mb-1">Perceived Priority</span>
                  <span className="font-semibold text-amber-400">{priority}</span>
                </div>
                <div>
                  <span className="text-gray-400 block mb-1">Attached Evidence</span>
                  <span className="font-semibold text-white">{files.length} file(s) attached</span>
                </div>
              </div>

              <div className="space-y-1">
                <span className="text-gray-400 block">Location Coordinates & Address</span>
                <div className="font-mono text-cyan-400">
                  {latitude.toFixed(6)}, {longitude.toFixed(6)}
                </div>
                <div className="text-gray-300">{locationAddress || 'No landmark specified'}</div>
              </div>
            </div>

            <div className="p-3.5 bg-blue-500/10 border border-blue-500/30 rounded-xl text-blue-300 text-xs leading-relaxed">
              ℹ Upon submission, an authoritative public reference (e.g. SC-2026-XXXXXX) will be generated. You will be able to track live municipal inspections, responses, and confirm resolution.
            </div>

            <div className="flex items-center justify-between pt-4 border-t border-gray-800">
              <button
                type="button"
                onClick={() => setStep(3)}
                className="px-5 py-2 text-xs text-gray-400 hover:text-white"
              >
                ← Back
              </button>
              <button
                type="button"
                disabled={isSubmitting}
                onClick={handleFinalSubmit}
                className="px-8 py-3 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white font-semibold text-xs rounded-xl transition shadow-lg shadow-emerald-600/30 flex items-center gap-2"
              >
                {isSubmitting ? (
                  <>
                    <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    <span>Filing Complaint into Database...</span>
                  </>
                ) : (
                  'Submit Complaint'
                )}
              </button>
            </div>
          </div>
        )}
      </div>
    </Layout>
  )
}
