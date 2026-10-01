import { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import Layout from '@/components/layout/Layout'
import {
  useAdminComplaintDetail,
  useUpdateComplaintStatus,
  useAssignComplaint,
  useAddPublicResponse,
  useAddInternalNote,
  useLinkHazard,
  useCreateHazard,
  useLinkRepair,
  useStaffUsers,
} from '@/hooks/useComplaintsApi'

export default function AdminComplaintDetail() {
  const { id } = useParams<{ id: string }>()
  const complaintId = Number(id)

  const { data, isLoading, error } = useAdminComplaintDetail(complaintId)
  const complaint = data?.complaint
  const aiCandidates = data?.ai_hazard_candidates || []
  const duplicates = data?.duplicate_candidates || []

  const { data: staffList = [] } = useStaffUsers()

  const updateStatusMutation = useUpdateComplaintStatus()
  const assignMutation = useAssignComplaint()
  const publicResponseMutation = useAddPublicResponse()
  const internalNoteMutation = useAddInternalNote()
  const linkHazardMutation = useLinkHazard()
  const createHazardMutation = useCreateHazard()
  const linkRepairMutation = useLinkRepair()

  // State forms
  const [statusComment, setStatusComment] = useState('')
  const [selectedStaffId, setSelectedStaffId] = useState<number | undefined>(undefined)
  const [department, setDepartment] = useState('Municipal Road Engineering')
  const [publicResponseText, setPublicResponseText] = useState('')
  const [internalNoteText, setInternalNoteText] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)

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

  async function handleStatusChange(newStatus: string) {
    setIsProcessing(true)
    try {
      await updateStatusMutation.mutateAsync({
        complaintId,
        new_status: newStatus,
        comment: statusComment.trim() || undefined,
      })
      setStatusComment('')
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Status transition failed.')
    } finally {
      setIsProcessing(false)
    }
  }

  async function handleAssign() {
    setIsProcessing(true)
    try {
      await assignMutation.mutateAsync({
        complaintId,
        assigned_to: selectedStaffId,
        assigned_department: department,
        notes: `Assigned via Command Center.`,
      })
      alert('Complaint assigned successfully.')
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Assignment failed.')
    } finally {
      setIsProcessing(false)
    }
  }

  async function handlePublicResponseSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!publicResponseText.trim()) return
    setIsProcessing(true)
    try {
      await publicResponseMutation.mutateAsync({
        complaintId,
        public_response: publicResponseText.trim(),
      })
      setPublicResponseText('')
      alert('Public response sent to citizen.')
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to post public response.')
    } finally {
      setIsProcessing(false)
    }
  }

  async function handleInternalNoteSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!internalNoteText.trim()) return
    setIsProcessing(true)
    try {
      await internalNoteMutation.mutateAsync({
        complaintId,
        internal_note: internalNoteText.trim(),
      })
      setInternalNoteText('')
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to add internal note.')
    } finally {
      setIsProcessing(false)
    }
  }

  async function handleLinkHazard(hazardId: number) {
    if (!confirm(`Link this complaint to AI Hazard #${hazardId}? This connects it to the SafeCity risk engine.`)) return
    setIsProcessing(true)
    try {
      await linkHazardMutation.mutateAsync({
        complaintId,
        hazard_id: hazardId,
        notes: 'Correlated by municipal officer in Operations Queue.',
      })
      alert(`Linked to Hazard #${hazardId} successfully!`)
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Linking failed.')
    } finally {
      setIsProcessing(false)
    }
  }

  async function handleCreateHazard() {
    if (!confirm('Promote this complaint to a new formal SafeCity Hazard on the nearest road segment?')) return
    setIsProcessing(true)
    try {
      await createHazardMutation.mutateAsync({
        complaintId,
        severity: complaint?.priority || 'HIGH',
        notes: 'Promoted from verified citizen complaint.',
      })
      alert('New SafeCity Hazard created and road risk recalculated!')
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Hazard promotion failed.')
    } finally {
      setIsProcessing(false)
    }
  }

  if (isLoading) {
    return (
      <Layout title="Operational Review" subtitle="Loading complaint...">
        <div className="py-20 text-center text-gray-500 text-xs">Loading operational dossier...</div>
      </Layout>
    )
  }

  if (error || !complaint) {
    return (
      <Layout title="Complaint Not Found" subtitle="Error">
        <div className="p-8 text-center text-red-400 text-xs">Failed to load complaint details.</div>
      </Layout>
    )
  }

  return (
    <Layout
      title={`Review Complaint: ${complaint.public_reference}`}
      subtitle={`Triage & Operational Action Queue · Filed ${new Date(complaint.created_at).toLocaleString()}`}
    >
      <div className="space-y-6">
        {/* Top Header Card */}
        <div className="bg-navy-800 border border-gray-800 rounded-2xl p-6 flex flex-col md:flex-row md:items-center justify-between gap-6 shadow-xl">
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
              <span className="text-xs font-semibold text-amber-400 px-2.5 py-0.5 rounded bg-amber-500/10 border border-amber-500/30">
                Priority: {complaint.priority}
              </span>
              <span className="text-xs text-gray-400">Category: {complaint.category.name}</span>
            </div>
            <h2 className="text-xl sm:text-2xl font-bold text-white tracking-tight">{complaint.title}</h2>
          </div>

          <Link
            to="/admin/complaints"
            className="self-start md:self-auto px-4 py-2 bg-navy-700 hover:bg-navy-600 border border-gray-700 text-xs text-white rounded-xl transition"
          >
            ← Back to All Complaints
          </Link>
        </div>

        {/* Possible Duplicate Alert Banner */}
        {duplicates.length > 0 && (
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-2xl p-4 sm:p-5 text-xs text-amber-300 space-y-2">
            <div className="flex items-center gap-2 font-bold text-amber-400">
              <span>⚠</span>
              <span>Potential Duplicate Complaints Detected ({duplicates.length})</span>
            </div>
            <p className="text-gray-300">
              The following active complaints exist in the same spatial vicinity ({duplicates[0].distance_meters}m away) with matching category.
            </p>
            <div className="flex flex-wrap gap-2 pt-1">
              {duplicates.map((d) => (
                <Link
                  key={d.complaint_id}
                  to={`/admin/complaints/${d.complaint_id}`}
                  className="px-3 py-1.5 bg-navy-900 border border-amber-500/40 rounded-lg hover:bg-navy-800 transition text-[11px] font-mono"
                >
                  {d.public_reference}: {d.title.slice(0, 30)}... ({d.distance_meters}m) →
                </Link>
              ))}
            </div>
          </div>
        )}

        {/* 3-Column Layout: Details (1/3), AI Correlation & Evidence (1/3), Actions & Notes (1/3) */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Column 1: Citizen Contact, Description, Location */}
          <div className="space-y-6">
            {/* Citizen Details Card */}
            <div className="bg-navy-800 border border-gray-800 rounded-2xl p-5 space-y-3">
              <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Citizen Information</h4>
              <div className="space-y-1.5 text-xs">
                <div className="text-white font-medium text-sm">{complaint.citizen_name}</div>
                <div className="text-gray-400 font-mono">{complaint.citizen_email}</div>
                {complaint.citizen_phone && (
                  <div className="text-gray-400 font-mono">{complaint.citizen_phone}</div>
                )}
                <div className="text-[10px] text-gray-500 pt-1">
                  Citizen Account ID: #{complaint.citizen_id}
                </div>
              </div>
            </div>

            {/* Description & Location Card */}
            <div className="bg-navy-800 border border-gray-800 rounded-2xl p-5 space-y-4 text-xs">
              <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Complaint Context</h4>

              <div className="space-y-1">
                <span className="text-gray-400 text-[11px]">Description</span>
                <p className="text-gray-200 leading-relaxed bg-navy-900/60 p-3 rounded-xl border border-gray-800">
                  {complaint.description}
                </p>
              </div>

              <div className="space-y-1">
                <span className="text-gray-400 text-[11px]">Location Coordinates</span>
                <div className="font-mono text-cyan-400 bg-navy-900/60 p-2.5 rounded-xl border border-gray-800">
                  {complaint.location_lat.toFixed(6)}, {complaint.location_lng.toFixed(6)}
                </div>
                <div className="text-gray-300 mt-1">{complaint.location_address || 'No landmark provided'}</div>
              </div>

              {/* Road Segment linkage */}
              {complaint.road_segment_id && (
                <div className="p-3 bg-navy-900 border border-gray-700/80 rounded-xl space-y-1">
                  <div className="text-[10px] text-gray-400 uppercase font-semibold">Matched Road Segment</div>
                  <div className="text-white font-medium">{complaint.road_name || `Road #${complaint.road_segment_id}`}</div>
                  <Link
                    to="/road-intelligence"
                    className="text-[11px] text-accent hover:underline block pt-0.5"
                  >
                    View Road Risk Dossier →
                  </Link>
                </div>
              )}
            </div>

            {/* Timeline Audit */}
            <div className="bg-navy-800 border border-gray-800 rounded-2xl p-5 space-y-4">
              <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Lifecycle History</h4>
              <div className="space-y-3">
                {complaint.status_history.map((h, i) => (
                  <div key={h.id || i} className="text-xs border-l-2 border-accent pl-3 space-y-0.5">
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-white">{h.new_status}</span>
                      <span className="text-[10px] text-gray-500 font-mono">{new Date(h.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
                    </div>
                    <p className="text-gray-400 text-[11px]">{h.comment}</p>
                    <div className="text-[10px] text-gray-500">By: {h.changed_by_name} ({h.changed_by_role})</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Column 2: Evidence & AI Hazard Correlation */}
          <div className="space-y-6">
            {/* Uploaded Evidence Gallery */}
            <div className="bg-navy-800 border border-gray-800 rounded-2xl p-5 space-y-4">
              <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
                Submitted Citizen Evidence ({complaint.evidence_items.length})
              </h4>

              {complaint.evidence_items.length === 0 ? (
                <div className="text-center py-6 text-gray-500 text-xs">No media files attached by citizen</div>
              ) : (
                <div className="space-y-3">
                  {complaint.evidence_items.map((ev) => (
                    <div key={ev.id} className="bg-navy-900 border border-gray-700 rounded-xl p-2.5 space-y-2">
                      {ev.file_type === 'video' ? (
                        <video src={ev.file_url} controls className="w-full h-40 object-cover rounded-lg bg-black" />
                      ) : (
                        <a href={ev.file_url} target="_blank" rel="noopener noreferrer">
                          <img
                            src={ev.file_url}
                            alt={ev.filename}
                            className="w-full h-40 object-cover rounded-lg hover:opacity-90 transition cursor-pointer"
                          />
                        </a>
                      )}
                      <div className="text-[11px] text-gray-300 font-mono truncate">{ev.filename}</div>
                      {ev.notes && <p className="text-[10px] text-gray-400 italic">{ev.notes}</p>}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* AI Hazard Correlation & SafeCity Loop Linking */}
            <div className="bg-navy-800 border border-gray-800 rounded-2xl p-5 space-y-4">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-semibold text-cyan-400 uppercase tracking-wider">
                  SafeCity AI Correlation
                </h4>
                {complaint.ai_linked && (
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-400 border border-cyan-500/40">
                    LINKED TO HAZARD #{complaint.hazard_id}
                  </span>
                )}
              </div>

              {aiCandidates.length === 0 ? (
                <div className="bg-navy-900 border border-gray-800 rounded-xl p-4 text-xs text-gray-400 space-y-2">
                  <p>No existing AI hazard observations within 300m threshold.</p>
                  <button
                    type="button"
                    disabled={isProcessing}
                    onClick={handleCreateHazard}
                    className="w-full py-2 bg-accent hover:bg-accent/90 disabled:opacity-50 text-white font-medium text-xs rounded-xl transition shadow-md shadow-accent/20"
                  >
                    Promote to New SafeCity Hazard
                  </button>
                </div>
              ) : (
                <div className="space-y-3">
                  <p className="text-xs text-gray-300">
                    Found {aiCandidates.length} AI candidate observation(s) within spatial threshold:
                  </p>
                  {aiCandidates.map((cand) => (
                    <div
                      key={cand.hazard_id}
                      className="bg-navy-900 border border-gray-700/80 rounded-xl p-3 space-y-2 text-xs"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-white">Hazard #{cand.hazard_id} ({cand.type})</span>
                        <span className="font-mono text-cyan-400 font-bold">
                          {Math.round(cand.confidence * 100)}% match
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-2 text-[11px] text-gray-400">
                        <div>Distance: <span className="text-white font-mono">{cand.distance_meters}m</span></div>
                        <div>Road: <span className="text-white">{cand.road_name}</span></div>
                        <div>Severity: <span className="text-amber-400">{cand.severity}</span></div>
                        <div>Risk: <span className="text-rose-400">{cand.risk_score}/100</span></div>
                      </div>

                      {cand.image_path && (
                        <div className="pt-1">
                          <img
                            src={cand.image_path}
                            alt="AI detection"
                            className="w-full h-24 object-cover rounded-lg border border-gray-800"
                          />
                        </div>
                      )}

                      <button
                        type="button"
                        disabled={isProcessing || complaint.hazard_id === cand.hazard_id}
                        onClick={() => handleLinkHazard(cand.hazard_id)}
                        className="w-full mt-1 py-1.5 bg-cyan-600/80 hover:bg-cyan-500 disabled:opacity-40 text-white font-medium text-xs rounded-lg transition"
                      >
                        {complaint.hazard_id === cand.hazard_id ? '✓ Already Linked' : 'Confirm Link to this Hazard'}
                      </button>
                    </div>
                  ))}

                  <button
                    type="button"
                    disabled={isProcessing}
                    onClick={handleCreateHazard}
                    className="w-full py-2 bg-navy-700 hover:bg-navy-600 text-gray-300 text-xs font-medium rounded-xl border border-gray-700 transition"
                  >
                    + Or Create Independent Hazard
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Column 3: Actions, Assignment, Public Response, Internal Notes */}
          <div className="space-y-6">
            {/* Status Workflow Actions */}
            <div className="bg-navy-800 border border-gray-800 rounded-2xl p-5 space-y-4">
              <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Workflow Transitions</h4>

              <div className="space-y-1.5">
                <label className="block text-[11px] text-gray-400">Transition Comment / Audit Reason</label>
                <input
                  type="text"
                  value={statusComment}
                  onChange={(e) => setStatusComment(e.target.value)}
                  placeholder="Optional rationale for audit log..."
                  className="w-full bg-navy-900 border border-gray-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-accent"
                />
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs font-medium">
                <button
                  type="button"
                  disabled={isProcessing}
                  onClick={() => handleStatusChange('UNDER_REVIEW')}
                  className="p-2.5 rounded-xl bg-amber-500/15 hover:bg-amber-500/25 border border-amber-500/40 text-amber-300 transition"
                >
                  Under Review
                </button>
                <button
                  type="button"
                  disabled={isProcessing}
                  onClick={() => handleStatusChange('VERIFIED')}
                  className="p-2.5 rounded-xl bg-indigo-500/15 hover:bg-indigo-500/25 border border-indigo-500/40 text-indigo-300 transition"
                >
                  Verify Defect
                </button>
                <button
                  type="button"
                  disabled={isProcessing}
                  onClick={() => handleStatusChange('IN_PROGRESS')}
                  className="p-2.5 rounded-xl bg-blue-500/15 hover:bg-blue-500/25 border border-blue-500/40 text-blue-300 transition"
                >
                  Start Work
                </button>
                <button
                  type="button"
                  disabled={isProcessing}
                  onClick={() => handleStatusChange('RESOLVED')}
                  className="p-2.5 rounded-xl bg-emerald-500/15 hover:bg-emerald-500/25 border border-emerald-500/40 text-emerald-300 transition"
                >
                  Mark Resolved
                </button>
                <button
                  type="button"
                  disabled={isProcessing}
                  onClick={() => handleStatusChange('REJECTED')}
                  className="p-2 rounded-xl bg-red-500/15 hover:bg-red-500/25 border border-red-500/40 text-red-300 transition text-[11px]"
                >
                  Reject
                </button>
                <button
                  type="button"
                  disabled={isProcessing}
                  onClick={() => handleStatusChange('CLOSED')}
                  className="p-2 rounded-xl bg-gray-700 hover:bg-gray-600 text-gray-200 transition text-[11px]"
                >
                  Close Case
                </button>
              </div>
            </div>

            {/* Staff Assignment */}
            <div className="bg-navy-800 border border-gray-800 rounded-2xl p-5 space-y-3">
              <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Assign Complaint</h4>

              <div className="space-y-2 text-xs">
                <div>
                  <label className="block text-gray-400 mb-1">Staff Member</label>
                  <select
                    value={selectedStaffId ?? complaint.assigned_to ?? ''}
                    onChange={(e) => setSelectedStaffId(e.target.value ? Number(e.target.value) : undefined)}
                    className="w-full bg-navy-900 border border-gray-700 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-accent"
                  >
                    <option value="">Select Staff Member</option>
                    {staffList.map((s) => (
                      <option key={s.id} value={s.id}>
                        {s.full_name} ({s.role})
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-gray-400 mb-1">Department</label>
                  <input
                    type="text"
                    value={department}
                    onChange={(e) => setDepartment(e.target.value)}
                    className="w-full bg-navy-900 border border-gray-700 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-accent"
                  />
                </div>

                <button
                  type="button"
                  disabled={isProcessing}
                  onClick={handleAssign}
                  className="w-full py-2 bg-navy-700 hover:bg-navy-600 border border-gray-700 text-white font-medium text-xs rounded-xl transition"
                >
                  Confirm Assignment
                </button>
              </div>
            </div>

            {/* Public Response Editor (Visible to citizen) */}
            <div className="bg-navy-800 border border-gray-800 rounded-2xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-semibold text-emerald-400 uppercase tracking-wider">
                  Public Response
                </h4>
                <span className="text-[10px] text-gray-500">Visible to Citizen</span>
              </div>

              {complaint.public_response && (
                <div className="p-3 bg-navy-900 border border-gray-700 rounded-xl text-xs text-gray-300">
                  <div className="text-[10px] text-gray-500 font-semibold mb-0.5">Current Response:</div>
                  {complaint.public_response}
                </div>
              )}

              <form onSubmit={handlePublicResponseSubmit} className="space-y-2">
                <textarea
                  rows={3}
                  required
                  value={publicResponseText}
                  onChange={(e) => setPublicResponseText(e.target.value)}
                  placeholder="Official municipal response to citizen..."
                  className="w-full bg-navy-900 border border-gray-700 rounded-xl p-3 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-accent"
                />
                <button
                  type="submit"
                  disabled={isProcessing}
                  className="w-full py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs rounded-xl transition shadow-md shadow-emerald-600/20"
                >
                  Send Response to Citizen
                </button>
              </form>
            </div>

            {/* Internal Notes (Confidential to staff) */}
            <div className="bg-navy-800 border border-gray-800 rounded-2xl p-5 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-semibold text-purple-400 uppercase tracking-wider">
                  Internal Notes
                </h4>
                <span className="text-[10px] text-rose-400 font-semibold">🔒 Staff Only</span>
              </div>

              {complaint.internal_notes && (
                <div className="p-3 bg-navy-900 border border-purple-500/20 rounded-xl text-xs text-purple-200 font-mono whitespace-pre-wrap max-h-48 overflow-y-auto">
                  {complaint.internal_notes}
                </div>
              )}

              <form onSubmit={handleInternalNoteSubmit} className="space-y-2">
                <textarea
                  rows={2}
                  required
                  value={internalNoteText}
                  onChange={(e) => setInternalNoteText(e.target.value)}
                  placeholder="Confidential notes, contractor details, cost estimates..."
                  className="w-full bg-navy-900 border border-gray-700 rounded-xl p-3 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
                />
                <button
                  type="submit"
                  disabled={isProcessing}
                  className="w-full py-2 bg-purple-600/80 hover:bg-purple-600 text-white font-medium text-xs rounded-xl transition"
                >
                  Append Internal Note
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  )
}
