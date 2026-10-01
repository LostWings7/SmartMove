import { useState } from 'react'
import { Link } from 'react-router-dom'
import Layout from '@/components/layout/Layout'
import { useAdminComplaints, useCategories, useStaffUsers } from '@/hooks/useComplaintsApi'

export default function AdminComplaints() {
  const [page, setPage] = useState(1)
  const [statusFilter, setStatusFilter] = useState<string>('')
  const [categoryFilter, setCategoryFilter] = useState<number | undefined>(undefined)
  const [priorityFilter, setPriorityFilter] = useState<string>('')
  const [assigneeFilter, setAssigneeFilter] = useState<number | undefined>(undefined)
  const [aiLinkedFilter, setAiLinkedFilter] = useState<boolean | undefined>(undefined)
  const [searchTerm, setSearchTerm] = useState('')
  const [sortBy, setSortBy] = useState('created_at')
  const [sortOrder, setSortOrder] = useState<'desc' | 'asc'>('desc')

  const { data: categories = [] } = useCategories()
  const { data: staffList = [] } = useStaffUsers()

  const { data, isLoading } = useAdminComplaints({
    page,
    page_size: 15,
    status: statusFilter || undefined,
    category_id: categoryFilter,
    priority: priorityFilter || undefined,
    assigned_to: assigneeFilter,
    ai_linked: aiLinkedFilter,
    search: searchTerm || undefined,
    sort_by: sortBy,
    sort_order: sortOrder,
  })

  const complaints = data?.items || []
  const totalPages = data?.total_pages || 1

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

  function handleExportCsv() {
    const params = new URLSearchParams()
    if (statusFilter) params.append('status', statusFilter)
    if (categoryFilter) params.append('category_id', String(categoryFilter))
    const token = localStorage.getItem('safecity_token')
    const exportUrl = `/api/admin/complaints/export?${params.toString()}`

    // Fetch and download
    fetch(exportUrl, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => res.blob())
      .then((blob) => {
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `SafeCity_Complaints_Report_${new Date().toISOString().slice(0, 10)}.csv`
        document.body.appendChild(a)
        a.click()
        a.remove()
      })
      .catch((err) => alert('Export failed: ' + err))
  }

  return (
    <Layout
      title="Municipal Complaints Operations Queue"
      subtitle="Search, triage, verify, assign, and resolve civic road complaints"
    >
      <div className="space-y-6">
        {/* Filter and Search Bar */}
        <div className="bg-navy-800 border border-gray-800 rounded-2xl p-5 space-y-4">
          <div className="flex flex-col lg:flex-row gap-3 items-stretch lg:items-center justify-between">
            {/* Search Input */}
            <div className="relative flex-1">
              <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-500 text-xs">🔍</span>
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => {
                  setSearchTerm(e.target.value)
                  setPage(1)
                }}
                placeholder="Search reference (SC-2026-000184), citizen name, title, address..."
                className="w-full bg-navy-900 border border-gray-700 rounded-xl pl-9 pr-4 py-2.5 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-accent"
              />
            </div>

            {/* Export CSV Button */}
            <button
              type="button"
              onClick={handleExportCsv}
              className="px-4 py-2 bg-navy-700 hover:bg-navy-600 border border-gray-700 text-white text-xs font-semibold rounded-xl transition flex items-center justify-center gap-2"
            >
              <span>📥</span>
              <span>Export CSV</span>
            </button>
          </div>

          {/* Secondary Filters Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 pt-1 text-xs">
            {/* Status Filter */}
            <select
              value={statusFilter}
              onChange={(e) => {
                setStatusFilter(e.target.value)
                setPage(1)
              }}
              className="bg-navy-900 border border-gray-700 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-accent"
            >
              <option value="">All Statuses</option>
              <option value="SUBMITTED">Submitted</option>
              <option value="UNDER_REVIEW">Under Review</option>
              <option value="VERIFIED">Verified</option>
              <option value="ASSIGNED">Assigned</option>
              <option value="IN_PROGRESS">In Progress</option>
              <option value="RESOLVED">Resolved</option>
              <option value="CLOSED">Closed</option>
              <option value="REOPENED">Reopened</option>
              <option value="REJECTED">Rejected</option>
            </select>

            {/* Category Filter */}
            <select
              value={categoryFilter ?? ''}
              onChange={(e) => {
                setCategoryFilter(e.target.value ? Number(e.target.value) : undefined)
                setPage(1)
              }}
              className="bg-navy-900 border border-gray-700 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-accent"
            >
              <option value="">All Categories</option>
              {categories.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>

            {/* Priority Filter */}
            <select
              value={priorityFilter}
              onChange={(e) => {
                setPriorityFilter(e.target.value)
                setPage(1)
              }}
              className="bg-navy-900 border border-gray-700 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-accent"
            >
              <option value="">All Priorities</option>
              <option value="LOW">Low</option>
              <option value="MEDIUM">Medium</option>
              <option value="HIGH">High</option>
              <option value="CRITICAL">Critical</option>
            </select>

            {/* Assignee Filter */}
            <select
              value={assigneeFilter ?? ''}
              onChange={(e) => {
                setAssigneeFilter(e.target.value ? Number(e.target.value) : undefined)
                setPage(1)
              }}
              className="bg-navy-900 border border-gray-700 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-accent"
            >
              <option value="">All Assignees</option>
              {staffList.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.full_name} ({s.role})
                </option>
              ))}
            </select>

            {/* AI-Linked Filter */}
            <select
              value={aiLinkedFilter === undefined ? '' : String(aiLinkedFilter)}
              onChange={(e) => {
                setAiLinkedFilter(e.target.value === '' ? undefined : e.target.value === 'true')
                setPage(1)
              }}
              className="bg-navy-900 border border-gray-700 rounded-xl px-3 py-2 text-white focus:outline-none focus:border-accent"
            >
              <option value="">All Sources</option>
              <option value="true">AI-Linked Only</option>
              <option value="false">Citizen-Only</option>
            </select>
          </div>
        </div>

        {/* Complaints Table */}
        <div className="bg-navy-800 border border-gray-800 rounded-2xl overflow-hidden shadow-xl">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-gray-800 text-gray-400 uppercase text-[10px] tracking-wider font-semibold bg-navy-900/60">
                  <th className="py-3 px-4">Reference</th>
                  <th className="py-3 px-4">Citizen</th>
                  <th className="py-3 px-4">Category</th>
                  <th className="py-3 px-4">Title / Address</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4">Priority</th>
                  <th className="py-3 px-4">Assigned Staff</th>
                  <th className="py-3 px-4">AI Correlated</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/60 text-gray-300">
                {isLoading ? (
                  <tr>
                    <td colSpan={9} className="py-12 text-center text-gray-500">
                      Loading complaints...
                    </td>
                  </tr>
                ) : complaints.length === 0 ? (
                  <tr>
                    <td colSpan={9} className="py-12 text-center text-gray-500">
                      No complaints match the specified filters.
                    </td>
                  </tr>
                ) : (
                  complaints.map((c) => (
                    <tr key={c.id} className="hover:bg-navy-700/40 transition">
                      <td className="py-3 px-4 font-mono font-bold text-accent whitespace-nowrap">
                        {c.public_reference}
                      </td>
                      <td className="py-3 px-4 whitespace-nowrap">
                        <div className="text-white font-medium">{c.citizen_name}</div>
                        <div className="text-[10px] text-gray-500 truncate max-w-[140px]">{c.citizen_email}</div>
                      </td>
                      <td className="py-3 px-4 whitespace-nowrap font-medium text-gray-200">
                        {c.category.name}
                      </td>
                      <td className="py-3 px-4 max-w-xs">
                        <div className="text-white font-medium truncate">{c.title}</div>
                        <div className="text-[10px] text-gray-500 truncate">
                          {c.location_address || `(${c.location_lat.toFixed(4)}, ${c.location_lng.toFixed(4)})`}
                        </div>
                      </td>
                      <td className="py-3 px-4 whitespace-nowrap">
                        <span
                          className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${
                            statusColors[c.status] || 'bg-gray-500/10 text-gray-400'
                          }`}
                        >
                          {c.status.replace(/_/g, ' ')}
                        </span>
                      </td>
                      <td className="py-3 px-4 whitespace-nowrap font-semibold">
                        <span
                          className={
                            c.priority === 'CRITICAL'
                              ? 'text-rose-400'
                              : c.priority === 'HIGH'
                              ? 'text-amber-400'
                              : 'text-gray-300'
                          }
                        >
                          {c.priority}
                        </span>
                      </td>
                      <td className="py-3 px-4 whitespace-nowrap">
                        {c.assigned_to_name ? (
                          <div className="text-white text-xs font-medium">{c.assigned_to_name}</div>
                        ) : (
                          <span className="text-gray-500 text-[11px] italic">Unassigned</span>
                        )}
                        {c.assigned_department && (
                          <div className="text-[10px] text-gray-500">{c.assigned_department}</div>
                        )}
                      </td>
                      <td className="py-3 px-4 whitespace-nowrap">
                        {c.ai_linked ? (
                          <span className="text-cyan-400 font-mono text-[10px] bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/30">
                            MATCHED ({Math.round((c.ai_confidence || 0.85) * 100)}%)
                          </span>
                        ) : (
                          <span className="text-gray-600 text-[11px]">—</span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-right whitespace-nowrap">
                        <Link
                          to={`/admin/complaints/${c.id}`}
                          className="px-3 py-1.5 bg-navy-700 hover:bg-navy-600 border border-gray-700 text-white rounded-lg transition font-medium"
                        >
                          Triage & Review →
                        </Link>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination Footer */}
          <div className="p-4 border-t border-gray-800 flex items-center justify-between text-xs text-gray-400 bg-navy-900/40">
            <div>
              Total: <span className="text-white font-semibold">{data?.total ?? 0}</span> complaints
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                disabled={page <= 1}
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                className="px-3 py-1 bg-navy-800 hover:bg-navy-700 disabled:opacity-40 border border-gray-700 rounded-lg text-white"
              >
                Previous
              </button>
              <span className="font-mono text-gray-300">
                Page {page} of {totalPages}
              </span>
              <button
                type="button"
                disabled={page >= totalPages}
                onClick={() => setPage((p) => p + 1)}
                className="px-3 py-1 bg-navy-800 hover:bg-navy-700 disabled:opacity-40 border border-gray-700 rounded-lg text-white"
              >
                Next
              </button>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  )
}
