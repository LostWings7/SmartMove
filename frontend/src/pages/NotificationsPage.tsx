import { Link } from 'react-router-dom'
import Layout from '@/components/layout/Layout'
import {
  useNotifications,
  useMarkNotificationRead,
  useMarkAllNotificationsRead,
} from '@/hooks/useComplaintsApi'

export default function NotificationsPage() {
  const { data: notifications = [], isLoading } = useNotifications()
  const markReadMutation = useMarkNotificationRead()
  const markAllReadMutation = useMarkAllNotificationsRead()

  const typeIcons: Record<string, string> = {
    status_update: '🔄',
    municipal_response: '🏛',
    assignment: '📋',
    complaint_submitted: '📥',
    complaint_reopened: '🚨',
    system: '⚙',
  }

  return (
    <Layout
      title="Notifications & Alerts"
      subtitle="Direct updates regarding your filed complaints, status changes, and municipal actions"
    >
      <div className="max-w-3xl mx-auto space-y-6">
        {/* Controls Bar */}
        <div className="bg-navy-800 border border-gray-800 rounded-2xl p-4 sm:p-5 flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-white">Your In-App Alerts</h3>
            <p className="text-gray-400 text-xs mt-0.5">Authoritative status updates synchronized in real time.</p>
          </div>

          <button
            type="button"
            disabled={notifications.filter((n) => !n.is_read).length === 0}
            onClick={() => markAllReadMutation.mutate()}
            className="px-3.5 py-2 bg-navy-700 hover:bg-navy-600 disabled:opacity-40 text-xs text-white font-medium rounded-xl border border-gray-700 transition"
          >
            Mark All as Read
          </button>
        </div>

        {/* Notifications List */}
        {isLoading ? (
          <div className="py-12 text-center text-gray-500 text-xs">Loading alerts...</div>
        ) : notifications.length === 0 ? (
          <div className="bg-navy-800 border border-gray-800 rounded-2xl p-12 text-center">
            <div className="text-3xl mb-2">🔔</div>
            <h4 className="text-white font-medium text-sm">No notifications right now</h4>
            <p className="text-gray-400 text-xs mt-1">
              You will receive notifications here whenever an officer reviews or resolves your complaints.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {notifications.map((n) => (
              <div
                key={n.id}
                className={`p-4 rounded-2xl border transition flex items-start gap-4 ${
                  n.is_read
                    ? 'bg-navy-800/70 border-gray-800/80 text-gray-400'
                    : 'bg-navy-800 border-accent/40 text-gray-200 shadow-md shadow-accent/5'
                }`}
              >
                <div className="text-2xl flex-shrink-0 mt-0.5">{typeIcons[n.type] || '🔔'}</div>

                <div className="flex-1 space-y-1">
                  <div className="flex items-center justify-between gap-2">
                    <h5 className="text-sm font-semibold text-white">{n.title}</h5>
                    <span className="text-[10px] text-gray-500 font-mono">
                      {new Date(n.created_at).toLocaleString()}
                    </span>
                  </div>

                  <p className="text-xs text-gray-300 leading-relaxed">{n.message}</p>

                  <div className="flex items-center gap-4 pt-2">
                    {n.related_complaint_id && (
                      <Link
                        to={`/citizen/complaints/${n.related_complaint_id}`}
                        className="text-xs text-accent hover:underline font-medium"
                      >
                        Inspect Complaint →
                      </Link>
                    )}
                    {!n.is_read && (
                      <button
                        type="button"
                        onClick={() => markReadMutation.mutate(n.id)}
                        className="text-[11px] text-gray-400 hover:text-white"
                      >
                        Mark as read
                      </button>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </Layout>
  )
}
