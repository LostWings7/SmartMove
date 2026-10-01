import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import Layout from '@/components/layout/Layout'
import { useComplaintDetail, useCitizenConfirmResolution, useUploadEvidence } from '@/hooks/useComplaintsApi'
import { voiceActionDispatcher } from '@/services/voiceActionDispatcher'

export default function ComplaintDetail() {
  const { id } = useParams<{ id: string }>()
  const complaintId = Number(id)

  const { data: complaint, isLoading, error } = useComplaintDetail(complaintId)
  const confirmMutation = useCitizenConfirmResolution()
  const uploadMutation = useUploadEvidence()

  // Citizen Confirmation / Reopen modal states
  const [showConfirmModal, setShowConfirmModal] = useState(false)
  const [confirmType, setConfirmType] = useState<'RESOLVE' | 'REOPEN'>('RESOLVE')
  const [confirmComment, setConfirmComment] = useState('')
  const [isSubmittingAction, setIsSubmittingAction] = useState(false)

  // Follow-up evidence upload state
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploadNotes, setUploadNotes] = useState('')
  const [isUploading, setIsUploading] = useState(false)

  // Live FRIDAY voice navigation within complaint
  useEffect(() => {
    const unsubHistory = voiceActionDispatcher.subscribe('show_status_history', () => {
      const el = document.getElementById('status-history')
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' })
        return true
      }
      return false
    })

    const unsubEvidence = voiceActionDispatcher.subscribe('open_evidence', () => {
      const el = document.getElementById('evidence-files')
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' })
        return true
      }
      return false
    })

    return () => {
      unsubHistory()
      unsubEvidence()
    }
  }, [])

  const statusColors: Record<string, string> = {
    SUBMITTED: 'bg-blue-500/10 text-blue-400 border-blue-500/30',
    ACKNOWLEDGED: 'bg-sky-500/10 text-sky-400 border-sky-500/30',
    UNDER_REVIEW: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    VERIFIED: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30',
    ASSIGNED: 'bg-purple-500/10 text-purple-400 border-purple-500/30',
    IN_PROGRESS: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30',
    RESOLVED: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    CITIZEN_CONFIRMATION: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    CLOSED: 'bg-gray-500/10 text-gray-400 border-gray-500/30',
    REJECTED: 'bg-red-500/10 text-red-400 border-red-500/30',
    REOPENED: 'bg-rose-500/10 text-rose-400 border-rose-500/30',
  }

  async function handleConfirmSubmit() {
    setIsSubmittingAction(true)
    try {
      await confirmMutation.mutateAsync({
        complaintId,
        confirmed: confirmType === 'RESOLVE',
        comment: confirmComment.trim() || undefined,
      })
      setShowConfirmModal(false)
      setConfirmComment('')
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to update complaint status.')
    } finally {
      setIsSubmittingAction(false)
    }
  }

  async function handleEvidenceUploadSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!uploadFile) return
    setIsUploading(true)
    try {
      await uploadMutation.mutateAsync({
        complaintId,
        file: uploadFile,
        notes: uploadNotes.trim() || 'Additional citizen evidence',
      })
      setUploadFile(null)
      setUploadNotes('')
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to upload evidence.')
    } finally {
      setIsUploading(false)
    }
  }

  if (isLoading) {
    return (
      <Layout title="Complaint Details" subtitle="Loading tracking information...">
        <div className="py-20 text-center text-gray-400 text-xs">Loading database record...</div>
      </Layout>
    )
  }

  if (error || !complaint) {
    return (
      <Layout title="Complaint Not Found" subtitle="Error retrieving complaint">
        <div className="bg-navy-800 border border-gray-800 rounded-2xl p-8 text-center max-w-md mx-auto">
          <div className="text-3xl mb-2">⚠</div>
          <h3 className="text-white font-semibold text-base">Complaint not accessible</h3>
          <p className="text-gray-400 text-xs mt-1">
            This complaint may not exist or belongs to another citizen under municipal privacy isolation.
          </p>
          <Link
            to="/citizen/complaints"
            className="mt-4 inline-block px-4 py-2 bg-accent text-white text-xs font-semibold rounded-xl"
          >
            ← Back to My Complaints
          </Link>
        </div>
      </Layout>
    )
  }

  return (
    <Layout
      title={`Complaint: ${complaint.public_reference}`}
      subtitle={`Filed ${new Date(complaint.created_at).toLocaleString()}`}
    >
      <div className="space-y-6">
        {/* Top Header Card */}
        <div className="bg-navy-800 border border-gray-800 rounded-2xl p-6 sm:p-8 flex flex-col md:flex-row md:items-center justify-between gap-6 shadow-xl">
          <div className="space-y-2">
            <div className="flex items-center gap-3 flex-wrap">
              <span className="font-mono text-sm font-bold text-accent bg-accent/10 border border-accent/20 px-3 py-1 rounded-lg">
                {complaint.public_reference}
              </span>
              <span
                className={`text-xs font-semibold px-3 py-1 rounded-full border ${
                  statusColors[complaint.status] || 'bg-gray-500/10 text-gray-400'
                }`}
              >
                {complaint.status.replace(/_/g, ' ')}
              </span>
              <span className="text-xs text-gray-400 font-medium">Category: {complaint.category.name}</span>
            </div>
            <h2 className="text-xl sm:text-2xl font-bold text-white tracking-tight">{complaint.title}</h2>
            <p className="text-gray-300 text-xs sm:text-sm max-w-2xl leading-relaxed">{complaint.description}</p>
          </div>

          {/* Quick Action / Confirm Resolution Button */}
          {complaint.can_confirm_resolution && (
            <div className="p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-2xl flex-shrink-0 flex flex-col gap-2">
              <div className="text-xs font-semibold text-emerald-400">Has this issue been resolved?</div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => {
                    setConfirmType('RESOLVE')
                    setShowConfirmModal(true)
                  }}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs rounded-xl shadow-md transition"
                >
                  ✓ Yes, Issue Resolved
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setConfirmType('REOPEN')
                    setShowConfirmModal(true)
                  }}
                  className="px-4 py-2 bg-rose-600/80 hover:bg-rose-500 text-white font-semibold text-xs rounded-xl transition"
                >
                  ✕ Problem Still Exists
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Official Municipal Response Card */}
        {complaint.public_response && (
          <div className="bg-emerald-950/30 border border-emerald-500/30 rounded-2xl p-6 space-y-2">
            <div className="flex items-center gap-2 text-emerald-400 text-xs font-semibold uppercase tracking-wider">
              <span>🏛</span>
              <span>Official Municipal Engineering Response</span>
            </div>
            <p className="text-gray-200 text-sm leading-relaxed">{complaint.public_response}</p>
            {complaint.resolved_at && (
              <div className="text-[11px] text-gray-400 font-mono pt-1">
                Resolution recorded: {new Date(complaint.resolved_at).toLocaleString()}
              </div>
            )}
          </div>
        )}

        {/* 2-Column Grid: Timeline on Left (2/3), Location & Evidence on Right (1/3) */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Vertical Status Timeline */}
          <div id="status-history" className="lg:col-span-2 bg-navy-800 border border-gray-800 rounded-2xl p-6 sm:p-8 space-y-6">
            <div className="flex items-center justify-between border-b border-gray-800 pb-4">
              <h3 className="text-base font-semibold text-white">Status Timeline & Audit History</h3>
              <span className="text-xs text-gray-400 font-mono">
                {complaint.status_history.length} lifecycle events
              </span>
            </div>

            <div className="relative pl-6 space-y-6 before:absolute before:left-2.5 before:top-2 before:bottom-2 before:w-0.5 before:bg-gray-800">
              {complaint.status_history.map((h, idx) => (
                <div key={h.id || idx} className="relative group">
                  {/* Timeline bullet */}
                  <div
                    className={`absolute -left-[27px] top-1 w-4 h-4 rounded-full border-2 bg-navy-900 ${
                      idx === complaint.status_history.length - 1
                        ? 'border-accent bg-accent/20 animate-pulse'
                        : 'border-emerald-500 bg-emerald-500/20'
                    }`}
                  />
                  <div className="bg-navy-900/60 border border-gray-800/80 rounded-xl p-4 space-y-1.5">
                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <span className="text-xs font-bold text-white tracking-wide">
                        {h.new_status.replace(/_/g, ' ')}
                      </span>
                      <span className="text-[10px] text-gray-400 font-mono">
                        {new Date(h.timestamp).toLocaleString()}
                      </span>
                    </div>

                    <div className="text-xs text-gray-300 leading-relaxed">{h.comment}</div>

                    <div className="text-[10px] text-gray-500 flex items-center gap-1.5 pt-1">
                      <span>Action by:</span>
                      <span className="font-semibold text-gray-400">{h.changed_by_name}</span>
                      <span className="font-mono text-gray-600">({h.changed_by_role})</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Right Column: Location Map & Uploaded Evidence */}
          <div className="space-y-6">
            {/* Location Card */}
            <div className="bg-navy-800 border border-gray-800 rounded-2xl p-5 space-y-3">
              <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Defect Location</h4>
              <div className="bg-navy-900 border border-gray-700 rounded-xl p-3 space-y-1">
                <div className="text-xs font-mono text-cyan-400">
                  {complaint.location_lat.toFixed(6)}, {complaint.location_lng.toFixed(6)}
                </div>
                <div className="text-xs text-gray-300">{complaint.location_address || 'Address not registered'}</div>
              </div>
              {complaint.assigned_department && (
                <div className="text-xs text-gray-400">
                  Assigned Team: <span className="text-white font-medium">{complaint.assigned_department}</span>
                </div>
              )}
            </div>

            {/* Submitted Evidence Gallery */}
            <div id="evidence-files" className="bg-navy-800 border border-gray-800 rounded-2xl p-5 space-y-4">
              <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                Evidence Files ({complaint.evidence_items.length})
              </h4>

              {complaint.evidence_items.length === 0 ? (
                <div className="text-center py-6 text-gray-500 text-xs">No media files attached</div>
              ) : (
                <div className="grid grid-cols-1 gap-3">
                  {complaint.evidence_items.map((ev) => (
                    <div
                      key={ev.id}
                      className="bg-navy-900 border border-gray-700/80 rounded-xl p-2.5 space-y-2 overflow-hidden"
                    >
                      {ev.file_type === 'video' ? (
                        <video src={ev.file_url} controls className="w-full h-36 object-cover rounded-lg bg-black" />
                      ) : (
                        <a href={ev.file_url} target="_blank" rel="noopener noreferrer">
                          <img
                            src={ev.file_url}
                            alt={ev.filename}
                            className="w-full h-36 object-cover rounded-lg hover:opacity-90 transition cursor-pointer"
                          />
                        </a>
                      )}
                      <div className="flex items-center justify-between text-[11px] text-gray-300">
                        <span className="truncate max-w-[160px] font-mono">{ev.filename}</span>
                        <span className="text-[10px] text-gray-500">{(ev.file_size / (1024 * 1024)).toFixed(2)} MB</span>
                      </div>
                      {ev.notes && <p className="text-[10px] text-gray-400 italic">{ev.notes}</p>}
                    </div>
                  ))}
                </div>
              )}

              {/* Add Follow-Up Evidence */}
              <form onSubmit={handleEvidenceUploadSubmit} className="pt-2 border-t border-gray-800 space-y-2">
                <label className="block text-[11px] font-medium text-gray-400">Add Follow-Up Evidence Photo</label>
                <input
                  type="file"
                  accept="image/*,video/*"
                  onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                  className="w-full text-xs text-gray-400 file:mr-2 file:py-1 file:px-2.5 file:rounded-lg file:border-0 file:text-xs file:bg-navy-700 file:text-white"
                />
                {uploadFile && (
                  <button
                    type="submit"
                    disabled={isUploading}
                    className="w-full py-1.5 bg-accent hover:bg-accent/90 disabled:opacity-50 text-white font-medium text-xs rounded-lg transition"
                  >
                    {isUploading ? 'Uploading...' : 'Upload Evidence'}
                  </button>
                )}
              </form>
            </div>
          </div>
        </div>
      </div>

      {/* Confirmation / Reopening Modal */}
      {showConfirmModal && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-navy-800 border border-gray-700 rounded-2xl p-6 max-w-md w-full space-y-4 shadow-2xl">
            <h3 className="text-base font-bold text-white">
              {confirmType === 'RESOLVE' ? 'Confirm Issue Resolution' : 'Report Issue Still Exists'}
            </h3>
            <p className="text-gray-300 text-xs leading-relaxed">
              {confirmType === 'RESOLVE'
                ? 'By confirming, this complaint will be transitioned to CLOSED status. Thank you for helping verify road safety repairs.'
                : 'If the road defect was not fixed satisfactorily, please provide notes so the engineering department can re-inspect.'}
            </p>

            <div>
              <label className="block text-xs font-medium text-gray-400 mb-1">
                {confirmType === 'RESOLVE' ? 'Resolution Comments (Optional)' : 'What problem still remains? *'}
              </label>
              <textarea
                rows={3}
                required={confirmType === 'REOPEN'}
                value={confirmComment}
                onChange={(e) => setConfirmComment(e.target.value)}
                placeholder={
                  confirmType === 'RESOLVE'
                    ? 'e.g. Inspected on 1 Oct, surface is smooth and safe.'
                    : 'e.g. Edge of the pothole is still sunken and damaged...'
                }
                className="w-full bg-navy-900 border border-gray-700 rounded-xl px-3 py-2 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-accent"
              />
            </div>

            <div className="flex items-center justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => setShowConfirmModal(false)}
                className="px-4 py-2 text-xs text-gray-400 hover:text-white"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={isSubmittingAction || (confirmType === 'REOPEN' && !confirmComment.trim())}
                onClick={handleConfirmSubmit}
                className={`px-5 py-2 rounded-xl text-xs font-semibold text-white transition ${
                  confirmType === 'RESOLVE'
                    ? 'bg-emerald-600 hover:bg-emerald-500'
                    : 'bg-rose-600 hover:bg-rose-500'
                }`}
              >
                {isSubmittingAction ? 'Updating...' : confirmType === 'RESOLVE' ? 'Confirm Resolved' : 'Reopen Complaint'}
              </button>
            </div>
          </div>
        </div>
      )}
    </Layout>
  )
}
