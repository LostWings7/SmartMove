import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '@/context/AuthContext'
import { useUnreadNotificationCount } from '@/hooks/useComplaintsApi'

interface TopBarProps {
  title: string
  subtitle?: string
  onToggleMobileMenu?: () => void
}

export default function TopBar({ title, subtitle, onToggleMobileMenu }: TopBarProps) {
  const { user, isAuthenticated, logout } = useAuth()
  const navigate = useNavigate()
  const { data: countData } = useUnreadNotificationCount()
  const unreadCount = countData?.unread_count || 0

  const now = new Date()
  const timeStr = now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })
  const dateStr = now.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' })

  const roleColors: Record<string, string> = {
    CITIZEN: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    INSPECTOR: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30',
    ENGINEER: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30',
    ADMIN: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    SUPERUSER: 'bg-purple-500/10 text-purple-400 border-purple-500/30',
  }

  return (
    <header className="h-14 bg-navy-800 border-b border-gray-800 flex items-center justify-between px-4 sm:px-6 flex-shrink-0 z-40">
      <div className="flex items-center gap-3">
        {onToggleMobileMenu && (
          <button
            onClick={onToggleMobileMenu}
            className="md:hidden text-gray-300 hover:text-white p-1 rounded hover:bg-navy-700 text-lg"
            aria-label="Open navigation menu"
          >
            ☰
          </button>
        )}
        <div>
          <h1 className="text-white font-semibold text-sm sm:text-base leading-none">{title}</h1>
          {subtitle && <p className="text-gray-500 text-[11px] sm:text-xs mt-0.5">{subtitle}</p>}
        </div>
      </div>

      <div className="flex items-center gap-3 sm:gap-4">
        {/* Notifications Icon with Badge */}
        {isAuthenticated && (
          <Link
            to={user?.role === 'CITIZEN' ? '/citizen/notifications' : '/citizen/notifications'}
            className="relative p-1.5 text-gray-400 hover:text-white rounded-lg hover:bg-navy-700 transition"
            title="In-App Notifications"
          >
            <span className="text-base">🔔</span>
            {unreadCount > 0 && (
              <span className="absolute -top-1 -right-1 bg-red-500 text-white text-[10px] font-bold px-1.5 py-0.2 rounded-full border border-navy-800">
                {unreadCount > 99 ? '99+' : unreadCount}
              </span>
            )}
          </Link>
        )}

        {/* User Identity / Role Badge */}
        {isAuthenticated && user ? (
          <div className="flex items-center gap-2 border-l border-gray-700 pl-3">
            <div className="hidden sm:block text-right">
              <div className="text-white text-xs font-semibold leading-tight">{user.full_name}</div>
              <div className="flex items-center justify-end gap-1.5 mt-0.5">
                <span
                  className={`text-[9px] font-mono px-1.5 py-0.2 rounded border font-semibold ${
                    roleColors[user.role] || 'bg-gray-500/10 text-gray-400 border-gray-500/30'
                  }`}
                >
                  {user.role}
                </span>
              </div>
            </div>
            <button
              onClick={logout}
              className="text-xs text-gray-400 hover:text-red-400 px-2 py-1 rounded hover:bg-navy-700 border border-gray-700 transition"
              title="Sign Out"
            >
              Sign Out
            </button>
          </div>
        ) : (
          <div className="flex items-center gap-2">
            <button
              onClick={() => navigate('/login')}
              className="text-xs bg-accent text-white px-3 py-1.5 rounded-lg hover:bg-accent/80 font-medium transition"
            >
              Sign In
            </button>
          </div>
        )}

        {/* Demo badge & Time */}
        <div className="hidden md:flex items-center gap-3 border-l border-gray-800 pl-3">
          <span className="demo-banner">⚠ DEMO DATA</span>
          <div className="text-right">
            <div className="text-white text-xs font-mono">{timeStr}</div>
            <div className="text-gray-500 text-[9px]">{dateStr}</div>
          </div>
        </div>
      </div>
    </header>
  )
}
