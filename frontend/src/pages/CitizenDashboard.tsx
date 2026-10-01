import { Link } from 'react-router-dom'
import Layout from '@/components/layout/Layout'
import StatusBadge from '@/components/ui/StatusBadge'
import { useCitizenDashboard } from '@/hooks/useComplaintsApi'

export default function CitizenDashboard() {
  const { data, isLoading, error } = useCitizenDashboard()

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

  return (
    <Layout
      title="Citizen Road Safety Portal"
      subtitle="Track your reports, view municipal resolutions, and file new road hazards"
    >
      <div className="space-y-6">
        {/* Welcome & Action Banner */}
        <div className="bg-gradient-to-r from-navy-800 via-navy-800 to-navy-700/60 border border-gray-800 rounded-2xl p-6 sm:p-8 flex flex-col sm:flex-row sm:items-center justify-between gap-6 shadow-xl relative overflow-hidden">
          <div className="relative z-10">
            <span className="text-[11px] font-semibold text-emerald-400 uppercase tracking-wider bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-0.5 rounded-full inline-block mb-2">
              Citizen Verified Portal
            </span>
            <h2 className="text-xl sm:text-2xl font-bold text-white tracking-tight">
              Welcome back, {data?.welcome_name || 'Citizen'}
            </h2>
            <p className="text-gray-400 text-xs sm:text-sm mt-1 max-w-xl">
              Thank you for contributing to safer streets. You can track status updates in real-time and confirm municipal repairs once completed.
            </p>
          </div>

          <div className="relative z-10 flex-shrink-0">
            <Link
              to="/citizen/complaints/new"
              className="inline-flex items-center gap-2 px-5 py-3 bg-accent hover:bg-accent/90 text-white font-medium text-sm rounded-xl transition shadow-lg shadow-accent/25 hover:scale-[1.02] active:scale-[0.98]"
            >
              <span className="text-base font-bold">＋</span>
              <span>File a Complaint</span>
            </Link>
          </div>
        </div>

        {/* Live Authoritative Database KPI Cards */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-navy-800 border border-gray-800 rounded-2xl p-4 sm:p-5">
            <div className="text-gray-400 text-xs font-medium">Active Complaints</div>
            <div className="text-2xl sm:text-3xl font-bold text-white mt-1">
              {isLoading ? '...' : data?.active_complaints ?? 0}
            </div>
            <div className="text-[11px] text-gray-500 mt-1">Under investigation or repair</div>
          </div>

          <div className="bg-navy-800 border border-gray-800 rounded-2xl p-4 sm:p-5">
            <div className="text-gray-400 text-xs font-medium">Pending Review</div>
            <div className="text-2xl sm:text-3xl font-bold text-amber-400 mt-1">
              {isLoading ? '...' : data?.pending_review ?? 0}
            </div>
            <div className="text-[11px] text-gray-500 mt-1">Awaiting field inspection</div>
          </div>

          <div className="bg-navy-800 border border-gray-800 rounded-2xl p-4 sm:p-5">
            <div className="text-gray-400 text-xs font-medium">In Progress</div>
            <div className="text-2xl sm:text-3xl font-bold text-blue-400 mt-1">
              {isLoading ? '...' : data?.in_progress ?? 0}
            </div>
            <div className="text-[11px] text-gray-500 mt-1">Engineering crew dispatched</div>
          </div>

          <div className="bg-navy-800 border border-gray-800 rounded-2xl p-4 sm:p-5">
            <div className="text-gray-400 text-xs font-medium">Resolved & Closed</div>
            <div className="text-2xl sm:text-3xl font-bold text-emerald-400 mt-1">
              {isLoading ? '...' : (data?.resolved ?? 0) + (data?.closed ?? 0)}
            </div>
            <div className="text-[11px] text-gray-500 mt-1">Completed civic actions</div>
          </div>
        </div>

        {/* Content Layout: Recent Complaints (2/3) + Notifications (1/3) */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Recent Complaints Column */}
          <div className="lg:col-span-2 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-white">Recent Complaints</h3>
              <Link to="/citizen/complaints" className="text-xs text-accent hover:underline font-medium">
                View All ({data?.total_complaints ?? 0}) →
              </Link>
            </div>

            {isLoading && (
              <div className="p-8 text-center text-gray-500 text-xs">Loading database records...</div>
            )}

            {!isLoading && (!data?.recent_complaints || data.recent_complaints.length === 0) && (
              <div className="bg-navy-800 border border-gray-800 rounded-2xl p-8 text-center">
                <div className="text-3xl mb-2">🛣</div>
                <div className="text-white font-medium text-sm">No complaints filed yet</div>
                <p className="text-gray-400 text-xs mt-1 max-w-sm mx-auto">
                  Notice a pothole, flooding, or damaged road infrastructure? Submit a report with photographic evidence.
                </p>
                <Link
                  to="/citizen/complaints/new"
                  className="mt-4 inline-block px-4 py-2 bg-accent text-white text-xs font-medium rounded-lg hover:bg-accent/90"
                >
                  File First Complaint
                </Link>
              </div>
            )}

            <div className="space-y-3">
              {data?.recent_complaints?.map((c) => (
                <div
                  key={c.id}
                  className="bg-navy-800 border border-gray-800 hover:border-gray-700 rounded-xl p-4 transition shadow-sm"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs font-semibold text-accent">{c.public_reference}</span>
                        <span
                          className={`text-[10px] font-semibold px-2 py-0.5 rounded-full border ${
                            statusColors[c.status] || 'bg-gray-500/10 text-gray-400'
                          }`}
                        >
                          {c.status.replace(/_/g, ' ')}
                        </span>
                        <span className="text-[11px] text-gray-400">· {c.category.name}</span>
                      </div>
                      <h4 className="text-white font-medium text-sm leading-snug">{c.title}</h4>
                      <p className="text-gray-400 text-xs line-clamp-2">{c.description}</p>
                    </div>

                    <Link
                      to={`/citizen/complaints/${c.id}`}
                      className="px-3 py-1.5 bg-navy-700 hover:bg-navy-600 text-white text-xs font-medium rounded-lg border border-gray-700 transition flex-shrink-0"
                    >
                      Track
                    </Link>
                  </div>

                  {/* Public municipal response preview if resolved */}
                  {c.public_response && (
                    <div className="mt-3 p-2.5 bg-navy-900/80 border border-gray-700/60 rounded-lg text-xs text-gray-300">
                      <span className="font-semibold text-emerald-400">Municipal Response: </span>
                      {c.public_response}
                    </div>
                  )}

                  <div className="mt-3 flex items-center justify-between text-[11px] text-gray-500 border-t border-gray-800/80 pt-2">
                    <span className="truncate max-w-[200px] sm:max-w-xs">📍 {c.location_address || `${c.location_lat.toFixed(4)}, ${c.location_lng.toFixed(4)}`}</span>
                    <span>{new Date(c.created_at).toLocaleDateString()}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Notifications Column */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-semibold text-white">Status Alerts</h3>
              <Link to="/citizen/notifications" className="text-xs text-accent hover:underline font-medium">
                All Alerts →
              </Link>
            </div>

            <div className="bg-navy-800 border border-gray-800 rounded-2xl p-4 space-y-3">
              {(!data?.recent_notifications || data.recent_notifications.length === 0) ? (
                <div className="text-center py-6 text-gray-500 text-xs">No new notifications</div>
              ) : (
                data.recent_notifications.map((n) => (
                  <div
                    key={n.id}
                    className={`p-3 rounded-xl border text-xs transition ${
                      n.is_read
                        ? 'bg-navy-900/40 border-gray-800/60 text-gray-400'
                        : 'bg-navy-900 border-accent/30 text-gray-200'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-semibold text-white">{n.title}</span>
                      <span className="text-[10px] text-gray-500 font-mono">
                        {new Date(n.created_at).toLocaleDateString()}
                      </span>
                    </div>
                    <p className="text-[11px] leading-relaxed">{n.message}</p>
                    {n.related_complaint_id && (
                      <Link
                        to={`/citizen/complaints/${n.related_complaint_id}`}
                        className="inline-block mt-2 text-[10px] text-accent hover:underline font-medium"
                      >
                        View Related Complaint →
                      </Link>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </Layout>
  )
}
