import { NavLink } from 'react-router-dom'
import { useAuth } from '@/context/AuthContext'

interface SidebarProps {
  mobileOpen?: boolean
  onCloseMobile?: () => void
}

export default function Sidebar({ mobileOpen = false, onCloseMobile }: SidebarProps) {
  const { user, isAuthenticated } = useAuth()
  const isCitizen = user?.role === 'CITIZEN'
  const isStaff = user && ['ADMIN', 'SUPERUSER', 'INSPECTOR', 'ENGINEER'].includes(user.role)
  const isAdminOrSuper = user && ['ADMIN', 'SUPERUSER'].includes(user.role)

  const citizenNavItems = [
    { path: '/citizen/dashboard', label: 'Dashboard', icon: '⬡' },
    { path: '/citizen/complaints', label: 'My Complaints', icon: '📋' },
    { path: '/citizen/complaints/new', label: 'File a Complaint', icon: '➕' },
    { path: '/citizen/notifications', label: 'Notifications', icon: '🔔' },
    { path: '/citizen/profile', label: 'My Profile', icon: '👤' },
  ]

  const citizenPublicItems = [
    { path: '/map', label: 'City Risk Map', icon: '🗺' },
    { path: '/routes', label: 'Safer Routes', icon: '↗' },
    { path: '/privacy', label: 'Privacy & Ethics', icon: '🛡' },
    { path: '/landing', label: 'SafeCity Showcase', icon: '🌟' },
  ]

  const staffDecisionItems = [
    { path: '/', label: 'Command Center', icon: '⬡' },
    { path: '/admin/complaints', label: 'Complaints Portal', icon: '📑' },
    { path: '/ai-vision', label: 'Vision AI Studio', icon: '👁' },
    { path: '/road-intelligence', label: 'Road Intelligence', icon: '🛣' },
    { path: '/traffic-conflicts', label: 'Traffic Conflicts', icon: '⚡' },
    { path: '/map', label: 'Risk Map', icon: '🗺' },
    { path: '/repair', label: 'Repair Priority', icon: '🔧' },
    ...(isAdminOrSuper ? [{ path: '/admin/users', label: 'User Accounts', icon: '👥' }] : []),
    { path: '/evidence', label: 'Evidence & Audit', icon: '📁' },
    { path: '/settings', label: 'Settings', icon: '⚙' },
  ]

  const staffUtilityItems = [
    { path: '/danger-zones', label: 'Danger Zones', icon: '⚠' },
    { path: '/digital-twin', label: 'Digital Twin', icon: '◈' },
    { path: '/junction-display', label: 'Roadside Billboard', icon: '🚨' },
    { path: '/analytics', label: 'City Analytics', icon: '📊' },
    { path: '/landing', label: 'Public Showcase', icon: '🌟' },
    { path: '/citizen/dashboard', label: 'Citizen View Preview', icon: '↗' },
  ]

  return (
    <>
      {/* Mobile overlay */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 md:hidden"
          onClick={onCloseMobile}
        />
      )}

      <aside
        className={`fixed left-0 top-0 h-full w-60 bg-navy-800 border-r border-gray-800 flex flex-col z-50 transition-transform duration-200 ${
          mobileOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        }`}
      >
        {/* Logo */}
        <div className="px-4 py-4 border-b border-gray-800 flex items-center justify-between">
          <NavLink to="/" onClick={onCloseMobile} className="flex items-center gap-2.5">
            <div className="w-7 h-7 bg-accent rounded-lg flex items-center justify-center text-xs font-black shadow-lg shadow-accent/20">
              SC
            </div>
            <div>
              <div className="text-white font-bold text-sm leading-none tracking-tight">SafeCity Loop</div>
              <div className="text-gray-400 text-[10px] mt-0.5 font-medium">GovTech Road Intelligence V2</div>
            </div>
          </NavLink>
          {onCloseMobile && (
            <button
              onClick={onCloseMobile}
              className="md:hidden text-gray-400 hover:text-white text-lg p-1 rounded hover:bg-navy-700"
              aria-label="Close navigation"
            >
              ✕
            </button>
          )}
        </div>

        {/* Navigation Sections */}
        <nav className="flex-1 overflow-y-auto px-2.5 py-3 space-y-4">
          {isCitizen ? (
            <>
              <div>
                <div className="px-2.5 mb-1.5 text-[10px] font-semibold text-emerald-400 uppercase tracking-wider">
                  Citizen Services
                </div>
                <div className="space-y-0.5">
                  {citizenNavItems.map((item) => (
                    <NavLink
                      key={item.path}
                      to={item.path}
                      end={item.path === '/citizen/dashboard'}
                      onClick={onCloseMobile}
                      className={({ isActive }) =>
                        isActive ? 'sidebar-item-active' : 'sidebar-item'
                      }
                    >
                      <span className="text-base w-5 text-center">{item.icon}</span>
                      <span className="truncate">{item.label}</span>
                    </NavLink>
                  ))}
                </div>
              </div>

              <div>
                <div className="px-2.5 mb-1.5 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">
                  City Navigation & Safety
                </div>
                <div className="space-y-0.5">
                  {citizenPublicItems.map((item) => (
                    <NavLink
                      key={item.path}
                      to={item.path}
                      onClick={onCloseMobile}
                      className={({ isActive }) =>
                        isActive ? 'sidebar-item-active text-xs' : 'sidebar-item text-xs text-gray-400'
                      }
                    >
                      <span className="text-sm w-5 text-center">{item.icon}</span>
                      <span className="truncate">{item.label}</span>
                    </NavLink>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <>
              <div>
                <div className="px-2.5 mb-1.5 text-[10px] font-semibold text-accent uppercase tracking-wider">
                  Decision Support
                </div>
                <div className="space-y-0.5">
                  {staffDecisionItems.map((item) => (
                    <NavLink
                      key={item.path}
                      to={item.path}
                      end={item.path === '/'}
                      onClick={onCloseMobile}
                      className={({ isActive }) =>
                        isActive ? 'sidebar-item-active' : 'sidebar-item'
                      }
                    >
                      <span className="text-base w-5 text-center">{item.icon}</span>
                      <span className="truncate">{item.label}</span>
                    </NavLink>
                  ))}
                </div>
              </div>

              <div>
                <div className="px-2.5 mb-1.5 text-[10px] font-semibold text-gray-500 uppercase tracking-wider">
                  GovTech Tools & Showcase
                </div>
                <div className="space-y-0.5">
                  {staffUtilityItems.map((item) => (
                    <NavLink
                      key={item.path}
                      to={item.path}
                      onClick={onCloseMobile}
                      className={({ isActive }) =>
                        isActive ? 'sidebar-item-active text-xs' : 'sidebar-item text-xs text-gray-400'
                      }
                    >
                      <span className="text-sm w-5 text-center">{item.icon}</span>
                      <span className="truncate">{item.label}</span>
                    </NavLink>
                  ))}
                </div>
              </div>
            </>
          )}
        </nav>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-gray-800 bg-navy-900/50">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-accent font-semibold tracking-wider uppercase">
              {isCitizen ? 'Citizen Portal' : 'Municipal Control'}
            </span>
            <span className="text-[9px] bg-amber-500/10 text-amber-400 border border-amber-500/30 px-1.5 py-0.2 rounded font-mono">
              [DEMO]
            </span>
          </div>
          <div className="text-[10px] text-gray-500 mt-0.5">Observe → Detect → Prioritize → Repair</div>
        </div>
      </aside>
    </>
  )
}
