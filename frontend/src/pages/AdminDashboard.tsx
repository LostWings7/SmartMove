import { Link } from 'react-router-dom'
import Layout from '@/components/layout/Layout'
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  CartesianGrid,
} from 'recharts'
import { useAdminDashboard } from '@/hooks/useComplaintsApi'

export default function AdminDashboard() {
  const { data, isLoading } = useAdminDashboard()

  const PIE_COLORS = ['#3b82f6', '#06b6d4', '#f59e0b', '#8b5cf6', '#10b981', '#ef4444']

  const statusData = data?.status_distribution
    ? Object.entries(data.status_distribution).map(([status, count]) => ({
        name: status.replace(/_/g, ' '),
        value: count,
      }))
    : []

  const categoryData = data?.category_distribution
    ? Object.entries(data.category_distribution).map(([category, count]) => ({
        category,
        count,
      }))
    : []

  return (
    <Layout
      title="Municipal Complaints Command Center"
      subtitle="Authoritative municipal complaint intelligence, operational throughput, and AI correlation"
    >
      <div className="space-y-6">
        {/* Top Operational Metrics Bar */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3.5">
          <div className="bg-navy-800 border border-gray-800 rounded-2xl p-4">
            <div className="text-[11px] font-medium text-gray-400">Total Complaints</div>
            <div className="text-2xl font-bold text-white mt-1">{isLoading ? '...' : data?.total_complaints ?? 0}</div>
            <div className="text-[10px] text-gray-500 mt-0.5">Authoritative DB</div>
          </div>

          <div className="bg-navy-800 border border-gray-800 rounded-2xl p-4">
            <div className="text-[11px] font-medium text-gray-400">New Today</div>
            <div className="text-2xl font-bold text-blue-400 mt-1">{isLoading ? '...' : data?.new_today ?? 0}</div>
            <div className="text-[10px] text-blue-500/70 mt-0.5">Inflow rate</div>
          </div>

          <div className="bg-navy-800 border border-gray-800 rounded-2xl p-4">
            <div className="text-[11px] font-medium text-gray-400">Under Review</div>
            <div className="text-2xl font-bold text-amber-400 mt-1">{isLoading ? '...' : data?.under_review ?? 0}</div>
            <div className="text-[10px] text-amber-500/70 mt-0.5">Awaiting inspection</div>
          </div>

          <div className="bg-navy-800 border border-gray-800 rounded-2xl p-4">
            <div className="text-[11px] font-medium text-gray-400">Assigned / In Progress</div>
            <div className="text-2xl font-bold text-purple-400 mt-1">
              {isLoading ? '...' : (data?.assigned ?? 0) + (data?.in_progress ?? 0)}
            </div>
            <div className="text-[10px] text-purple-500/70 mt-0.5">Active work orders</div>
          </div>

          <div className="bg-navy-800 border border-gray-800 rounded-2xl p-4">
            <div className="text-[11px] font-medium text-gray-400">Resolved & Closed</div>
            <div className="text-2xl font-bold text-emerald-400 mt-1">
              {isLoading ? '...' : (data?.resolved ?? 0) + (data?.closed ?? 0)}
            </div>
            <div className="text-[10px] text-emerald-500/70 mt-0.5">Civic resolutions</div>
          </div>

          <div className="bg-navy-800 border border-gray-800 rounded-2xl p-4">
            <div className="text-[11px] font-medium text-gray-400">AI-Correlated</div>
            <div className="text-2xl font-bold text-cyan-400 mt-1">{isLoading ? '...' : data?.ai_linked_count ?? 0}</div>
            <div className="text-[10px] text-cyan-500/70 mt-0.5">Linked to AI hazards</div>
          </div>
        </div>

        {/* Secondary KPI Bar: Resolution Speed & Citizen Confirmation */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="bg-navy-800/90 border border-gray-800 rounded-2xl p-4 flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-400 font-medium">Avg Resolution Time</div>
              <div className="text-lg font-bold text-white mt-0.5">
                {isLoading ? '...' : `${data?.avg_resolution_hours ?? 0} hours`}
              </div>
            </div>
            <span className="text-2xl">⏱</span>
          </div>

          <div className="bg-navy-800/90 border border-gray-800 rounded-2xl p-4 flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-400 font-medium">Citizen-Confirmed Resolutions</div>
              <div className="text-lg font-bold text-emerald-400 mt-0.5">
                {isLoading ? '...' : `${data?.citizen_confirmed_count ?? 0} verified`}
              </div>
            </div>
            <span className="text-2xl">✓</span>
          </div>

          <div className="bg-navy-800/90 border border-gray-800 rounded-2xl p-4 flex items-center justify-between">
            <div>
              <div className="text-xs text-gray-400 font-medium">Reopened Rate</div>
              <div className="text-lg font-bold text-rose-400 mt-0.5">
                {isLoading ? '...' : `${data?.reopened ?? 0} cases`}
              </div>
            </div>
            <span className="text-2xl">🔁</span>
          </div>
        </div>

        {/* Charts Grid: Real Recharts from SQLite Aggregations */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Chart 1: 7-Day Inflow vs Resolution Trend */}
          <div className="lg:col-span-2 bg-navy-800 border border-gray-800 rounded-2xl p-5 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-semibold text-white">7-Day Civic Throughput Trend</h4>
                <p className="text-[11px] text-gray-400">Submissions vs Completed Municipal Resolutions</p>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-navy-900 border border-gray-700 text-gray-300">
                Live DB
              </span>
            </div>

            <div className="h-[260px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data?.timeline_trend || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="date" stroke="#64748b" fontSize={11} />
                  <YAxis stroke="#64748b" fontSize={11} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', fontSize: '12px' }}
                  />
                  <Line type="monotone" dataKey="submitted" stroke="#3b82f6" strokeWidth={2} name="Submitted" />
                  <Line type="monotone" dataKey="resolved" stroke="#10b981" strokeWidth={2} name="Resolved" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Chart 2: Status Distribution Pie */}
          <div className="bg-navy-800 border border-gray-800 rounded-2xl p-5 space-y-3">
            <h4 className="text-sm font-semibold text-white">Status Breakdown</h4>
            <div className="h-[220px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={statusData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {statusData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '11px' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex flex-wrap gap-2 text-[10px] text-gray-400 justify-center">
              {statusData.map((s, idx) => (
                <div key={s.name} className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: PIE_COLORS[idx % PIE_COLORS.length] }} />
                  <span>{s.name}: {s.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Chart 3: Category Distribution Bar Chart */}
        <div className="bg-navy-800 border border-gray-800 rounded-2xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-white">Complaints by Category</h4>
            <Link to="/admin/complaints" className="text-xs text-accent hover:underline font-medium">
              View All Complaints Table →
            </Link>
          </div>

          <div className="h-[200px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categoryData} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="category" stroke="#64748b" fontSize={10} />
                <YAxis stroke="#64748b" fontSize={11} allowDecimals={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '12px', fontSize: '12px' }}
                />
                <Bar dataKey="count" fill="#38bdf8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Recent Operational Inflow Table */}
        <div className="bg-navy-800 border border-gray-800 rounded-2xl p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-white">Recent Complaints Inflow</h4>
            <Link to="/admin/complaints" className="text-xs text-accent hover:underline font-medium">
              Open Full Operations Queue →
            </Link>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-gray-800 text-gray-400 uppercase text-[10px] tracking-wider font-semibold">
                  <th className="py-2.5 px-3">Reference</th>
                  <th className="py-2.5 px-3">Citizen</th>
                  <th className="py-2.5 px-3">Category</th>
                  <th className="py-2.5 px-3">Title</th>
                  <th className="py-2.5 px-3">Status</th>
                  <th className="py-2.5 px-3">Priority</th>
                  <th className="py-2.5 px-3">AI Linked</th>
                  <th className="py-2.5 px-3 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/60 text-gray-300">
                {data?.recent_complaints?.map((c) => (
                  <tr key={c.id} className="hover:bg-navy-700/40 transition">
                    <td className="py-2.5 px-3 font-mono font-bold text-accent">{c.public_reference}</td>
                    <td className="py-2.5 px-3 text-white font-medium">{c.citizen_name}</td>
                    <td className="py-2.5 px-3">{c.category.name}</td>
                    <td className="py-2.5 px-3 truncate max-w-xs">{c.title}</td>
                    <td className="py-2.5 px-3">
                      <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-navy-900 border border-gray-700 text-gray-200">
                        {c.status.replace(/_/g, ' ')}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 font-semibold text-amber-400">{c.priority}</td>
                    <td className="py-2.5 px-3">
                      {c.ai_linked ? (
                        <span className="text-cyan-400 font-mono text-[10px] bg-cyan-500/10 px-1.5 py-0.5 rounded border border-cyan-500/30">
                          MATCHED
                        </span>
                      ) : (
                        <span className="text-gray-500 text-[10px]">—</span>
                      )}
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <Link
                        to={`/admin/complaints/${c.id}`}
                        className="px-2.5 py-1 bg-navy-700 hover:bg-navy-600 text-white rounded-lg border border-gray-700 transition"
                      >
                        Inspect
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </Layout>
  )
}
