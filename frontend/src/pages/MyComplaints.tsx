import { useState, useEffect } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import Layout from '@/components/layout/Layout'
import { useMyComplaints, useCategories } from '@/hooks/useComplaintsApi'

export default function MyComplaints() {
  const [searchParams] = useSearchParams()
  const [statusFilter, setStatusFilter] = useState<string>('')
  const [categoryFilter, setCategoryFilter] = useState<number | undefined>(undefined)
  const [searchTerm, setSearchTerm] = useState('')
  const [sortOrder, setSortOrder] = useState<'desc' | 'asc'>('desc')

  useEffect(() => {
    const ref = searchParams.get('reference') || searchParams.get('search')
    if (ref) setSearchTerm(ref)
  }, [searchParams])

  const { data: categories = [] } = useCategories()
  const { data: complaints = [], isLoading } = useMyComplaints({
    status: statusFilter || undefined,
    category_id: categoryFilter,
    search: searchTerm || undefined,
    sort_order: sortOrder,
  })

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

  const statusFilters = [
    { label: 'All', value: '' },
    { label: 'Under Review', value: 'UNDER_REVIEW' },
    { label: 'In Progress', value: 'IN_PROGRESS' },
    { label: 'Resolved', value: 'RESOLVED' },
    { label: 'Closed', value: 'CLOSED' },
  ]

  return (
    <Layout
      title="My Complaints"
      subtitle="View the authoritative lifecycle, timeline, and municipal responses for your reported issues"
    >
      <div className="space-y-6">
        {/* Top Controls Toolbar */}
        <div className="bg-navy-800 border border-gray-800 rounded-2xl p-4 sm:p-5 space-y-4">
          <div className="flex flex-col sm:flex-row gap-3 items-stretch sm:items-center justify-between">
            {/* Search Input */}
            <div className="relative flex-1">
              <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-500 text-xs">🔍</span>
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search reference (e.g. SC-2026-000184) or title..."
                className="w-full bg-navy-900 border border-gray-700 rounded-xl pl-9 pr-4 py-2 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-accent"
              />
            </div>

            {/* Category Dropdown */}
            <select
              value={categoryFilter ?? ''}
              onChange={(e) => setCategoryFilter(e.target.value ? Number(e.target.value) : undefined)}
              className="bg-navy-900 border border-gray-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-accent"
            >
              <option value="">All Categories</option>
              {categories.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>

            {/* Sort Toggle */}
            <button
              onClick={() => setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc')}
              className="px-3.5 py-2 bg-navy-900 border border-gray-700 hover:border-gray-600 rounded-xl text-xs text-gray-300 font-mono transition"
            >
              Sort: {sortOrder === 'desc' ? 'Newest ↓' : 'Oldest ↑'}
            </button>

            {/* File New CTA */}
            <Link
              to="/citizen/complaints/new"
              className="px-4 py-2 bg-accent hover:bg-accent/90 text-white font-medium text-xs rounded-xl transition text-center shadow-md shadow-accent/20"
            >
              + File Complaint
            </Link>
          </div>

          {/* Status Filter Pills */}
          <div className="flex items-center gap-2 overflow-x-auto pb-1">
            {statusFilters.map((sf) => (
              <button
                key={sf.value}
                onClick={() => setStatusFilter(sf.value)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition ${
                  statusFilter === sf.value
                    ? 'bg-accent text-white shadow-sm'
                    : 'bg-navy-900 text-gray-400 hover:text-white border border-gray-800'
                }`}
              >
                {sf.label}
              </button>
            ))}
          </div>
        </div>

        {/* Complaints Listing */}
        {isLoading ? (
          <div className="py-12 text-center text-gray-500 text-xs">Loading complaints from database...</div>
        ) : complaints.length === 0 ? (
          <div className="bg-navy-800 border border-gray-800 rounded-2xl p-12 text-center">
            <div className="text-4xl mb-3">📋</div>
            <h3 className="text-white font-medium text-base">No complaints match your filters</h3>
            <p className="text-gray-400 text-xs mt-1 max-w-sm mx-auto">
              Try adjusting your search query, or submit a new complaint to alert municipal teams.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {complaints.map((c) => (
              <div
                key={c.id}
                className="bg-navy-800 border border-gray-800 hover:border-gray-700 rounded-2xl p-5 transition shadow-sm space-y-3"
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-gray-800/80 pb-3">
                  <div className="flex items-center gap-2.5 flex-wrap">
                    <span className="font-mono text-xs font-bold text-accent bg-accent/10 border border-accent/20 px-2.5 py-0.5 rounded">
                      {c.public_reference}
                    </span>
                    <span
                      className={`text-[10px] font-semibold px-2.5 py-0.5 rounded-full border ${
                        statusColors[c.status] || 'bg-gray-500/10 text-gray-400'
                      }`}
                    >
                      {c.status.replace(/_/g, ' ')}
                    </span>
                    <span className="text-xs text-gray-300 font-medium">· {c.category.name}</span>
                    <span className="text-[11px] text-gray-500">
                      Filed on {new Date(c.created_at).toLocaleDateString()}
                    </span>
                  </div>

                  <Link
                    to={`/citizen/complaints/${c.id}`}
                    className="self-start sm:self-auto px-4 py-1.5 bg-navy-700 hover:bg-navy-600 border border-gray-700 text-white text-xs font-semibold rounded-xl transition"
                  >
                    View Details & Timeline →
                  </Link>
                </div>

                <div className="space-y-1">
                  <h4 className="text-base font-semibold text-white">{c.title}</h4>
                  <p className="text-gray-400 text-xs line-clamp-2 leading-relaxed">{c.description}</p>
                </div>

                {/* Evidence thumbnails */}
                {c.evidence_items && c.evidence_items.length > 0 && (
                  <div className="flex items-center gap-2 pt-1">
                    <span className="text-[11px] text-gray-500">Attached:</span>
                    {c.evidence_items.map((ev) => (
                      <span
                        key={ev.id}
                        className="text-[10px] font-mono px-2 py-0.5 rounded bg-navy-900 border border-gray-700 text-gray-300"
                      >
                        {ev.file_type === 'video' ? '🎥 Video' : '🖼 Photo'} ({ev.filename})
                      </span>
                    ))}
                  </div>
                )}

                {/* Municipal Response banner */}
                {c.public_response && (
                  <div className="p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-xs text-gray-300">
                    <div className="font-semibold text-emerald-400 mb-0.5">Municipal Response:</div>
                    <p className="line-clamp-2">{c.public_response}</p>
                  </div>
                )}

                <div className="flex items-center justify-between text-[11px] text-gray-500 pt-1">
                  <span className="truncate max-w-md">
                    📍 {c.location_address || `Lat ${c.location_lat.toFixed(4)}, Lng ${c.location_lng.toFixed(4)}`}
                  </span>
                  {c.assigned_department && (
                    <span className="text-gray-400 font-medium">Dept: {c.assigned_department}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </Layout>
  )
}
