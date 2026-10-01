import { useState } from 'react'
import Layout from '@/components/layout/Layout'
import {
  useAdminUsers,
  useUpdateUserRole,
  useUpdateUserStatus,
} from '@/hooks/useComplaintsApi'
import type { User, UserRole } from '@/types'

export default function UserManagement() {
  const [page, setPage] = useState(1)
  const [roleFilter, setRoleFilter] = useState<string>('')
  const [activeFilter, setActiveFilter] = useState<boolean | undefined>(undefined)
  const [searchTerm, setSearchTerm] = useState('')

  const { data, isLoading } = useAdminUsers({
    page,
    page_size: 15,
    role: roleFilter || undefined,
    is_active: activeFilter,
    search: searchTerm || undefined,
  })

  const updateRoleMutation = useUpdateUserRole()
  const updateStatusMutation = useUpdateUserStatus()

  // Change Role Modal
  const [selectedUser, setSelectedUser] = useState<User | null>(null)
  const [targetRole, setTargetRole] = useState<UserRole>('CITIZEN')
  const [modalError, setModalError] = useState<string | null>(null)
  const [isUpdating, setIsUpdating] = useState(false)

  const users = data?.items || []
  const totalPages = data?.total_pages || 1

  const roleColors: Record<string, string> = {
    CITIZEN: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    INSPECTOR: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/30',
    ENGINEER: 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30',
    ADMIN: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
    SUPERUSER: 'bg-purple-500/10 text-purple-400 border-purple-500/30',
  }

  async function handleRoleChangeSubmit() {
    if (!selectedUser) return
    setIsUpdating(true)
    setModalError(null)
    try {
      await updateRoleMutation.mutateAsync({
        userId: selectedUser.id,
        role: targetRole,
      })
      setSelectedUser(null)
    } catch (err: any) {
      setModalError(err.response?.data?.detail || 'Failed to update user role.')
    } finally {
      setIsUpdating(false)
    }
  }

  async function handleToggleStatus(user: User) {
    if (
      !confirm(
        `Are you sure you want to ${user.is_active ? 'deactivate' : 'activate'} account for ${user.full_name}?`
      )
    ) {
      return
    }
    try {
      await updateStatusMutation.mutateAsync({
        userId: user.id,
        is_active: !user.is_active,
      })
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to update user status.')
    }
  }

  return (
    <Layout
      title="User Accounts & Permissions"
      subtitle="Role-based access control, account lifecycle, and last-superuser protection"
    >
      <div className="space-y-6">
        {/* Filter Bar */}
        <div className="bg-navy-800 border border-gray-800 rounded-2xl p-5 flex flex-col sm:flex-row gap-3 items-stretch sm:items-center justify-between">
          <div className="relative flex-1">
            <span className="absolute left-3.5 top-1/2 -translate-y-1/2 text-gray-500 text-xs">🔍</span>
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => {
                setSearchTerm(e.target.value)
                setPage(1)
              }}
              placeholder="Search by name, email, or phone..."
              className="w-full bg-navy-900 border border-gray-700 rounded-xl pl-9 pr-4 py-2.5 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-accent"
            />
          </div>

          <div className="flex items-center gap-3">
            <select
              value={roleFilter}
              onChange={(e) => {
                setRoleFilter(e.target.value)
                setPage(1)
              }}
              className="bg-navy-900 border border-gray-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-accent"
            >
              <option value="">All Roles</option>
              <option value="CITIZEN">Citizen</option>
              <option value="INSPECTOR">Inspector</option>
              <option value="ENGINEER">Engineer</option>
              <option value="ADMIN">Admin</option>
              <option value="SUPERUSER">Superuser</option>
            </select>

            <select
              value={activeFilter === undefined ? '' : String(activeFilter)}
              onChange={(e) => {
                setActiveFilter(e.target.value === '' ? undefined : e.target.value === 'true')
                setPage(1)
              }}
              className="bg-navy-900 border border-gray-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-accent"
            >
              <option value="">All Statuses</option>
              <option value="true">Active Only</option>
              <option value="false">Deactivated Only</option>
            </select>
          </div>
        </div>

        {/* Users Table */}
        <div className="bg-navy-800 border border-gray-800 rounded-2xl overflow-hidden shadow-xl">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-gray-800 text-gray-400 uppercase text-[10px] tracking-wider font-semibold bg-navy-900/60">
                  <th className="py-3 px-4">User</th>
                  <th className="py-3 px-4">Email</th>
                  <th className="py-3 px-4">Phone</th>
                  <th className="py-3 px-4">Role</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4">Created Date</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/60 text-gray-300">
                {isLoading ? (
                  <tr>
                    <td colSpan={7} className="py-12 text-center text-gray-500">
                      Loading users...
                    </td>
                  </tr>
                ) : users.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="py-12 text-center text-gray-500">
                      No user accounts found matching filters.
                    </td>
                  </tr>
                ) : (
                  users.map((u) => (
                    <tr key={u.id} className="hover:bg-navy-700/40 transition">
                      <td className="py-3 px-4 font-medium text-white whitespace-nowrap">
                        {u.full_name}
                      </td>
                      <td className="py-3 px-4 font-mono text-gray-300 whitespace-nowrap">{u.email}</td>
                      <td className="py-3 px-4 font-mono text-gray-400 whitespace-nowrap">
                        {u.phone_number || '—'}
                      </td>
                      <td className="py-3 px-4 whitespace-nowrap">
                        <span
                          className={`text-[10px] font-mono font-semibold px-2 py-0.5 rounded border ${
                            roleColors[u.role] || 'bg-gray-500/10 text-gray-400'
                          }`}
                        >
                          {u.role}
                        </span>
                      </td>
                      <td className="py-3 px-4 whitespace-nowrap">
                        {u.is_active ? (
                          <span className="text-emerald-400 text-[11px] font-medium flex items-center gap-1.5">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                            Active
                          </span>
                        ) : (
                          <span className="text-red-400 text-[11px] font-medium flex items-center gap-1.5">
                            <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
                            Suspended
                          </span>
                        )}
                      </td>
                      <td className="py-3 px-4 text-gray-500 font-mono text-[11px] whitespace-nowrap">
                        {new Date(u.created_at).toLocaleDateString()}
                      </td>
                      <td className="py-3 px-4 text-right space-x-2 whitespace-nowrap">
                        <button
                          type="button"
                          onClick={() => {
                            setSelectedUser(u)
                            setTargetRole(u.role)
                            setModalError(null)
                          }}
                          className="px-2.5 py-1 bg-navy-700 hover:bg-navy-600 text-white rounded-lg border border-gray-700 transition"
                        >
                          Change Role
                        </button>
                        <button
                          type="button"
                          onClick={() => handleToggleStatus(u)}
                          className={`px-2.5 py-1 rounded-lg border text-xs transition ${
                            u.is_active
                              ? 'border-red-500/40 text-red-400 hover:bg-red-500/10'
                              : 'border-emerald-500/40 text-emerald-400 hover:bg-emerald-500/10'
                          }`}
                        >
                          {u.is_active ? 'Deactivate' : 'Activate'}
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          <div className="p-4 border-t border-gray-800 flex items-center justify-between text-xs text-gray-400 bg-navy-900/40">
            <div>
              Total: <span className="text-white font-semibold">{data?.total ?? 0}</span> accounts
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

      {/* Change Role Modal */}
      {selectedUser && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-navy-800 border border-gray-700 rounded-2xl p-6 max-w-sm w-full space-y-4 shadow-2xl">
            <h3 className="text-base font-bold text-white">Change Account Role</h3>
            <p className="text-gray-300 text-xs">
              Assign a new municipal privilege role to{' '}
              <span className="text-white font-semibold">{selectedUser.full_name}</span>.
            </p>

            {modalError && (
              <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-xs">
                ⚠ {modalError}
              </div>
            )}

            <div>
              <label className="block text-xs font-medium text-gray-400 mb-1">New Role</label>
              <select
                value={targetRole}
                onChange={(e) => setTargetRole(e.target.value as UserRole)}
                className="w-full bg-navy-900 border border-gray-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-accent"
              >
                <option value="CITIZEN">Citizen</option>
                <option value="INSPECTOR">Inspector</option>
                <option value="ENGINEER">Engineer</option>
                <option value="ADMIN">Admin</option>
                <option value="SUPERUSER">Superuser</option>
              </select>
            </div>

            <div className="text-[11px] text-gray-400 leading-tight">
              Note: Demoting or deactivating the last active superuser is strictly prevented by system safety rules.
            </div>

            <div className="flex items-center justify-end gap-2 pt-2">
              <button
                type="button"
                onClick={() => setSelectedUser(null)}
                className="px-4 py-2 text-xs text-gray-400 hover:text-white"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={isUpdating}
                onClick={handleRoleChangeSubmit}
                className="px-5 py-2 bg-accent hover:bg-accent/90 disabled:opacity-50 text-white text-xs font-semibold rounded-xl transition"
              >
                {isUpdating ? 'Saving...' : 'Apply Role'}
              </button>
            </div>
          </div>
        </div>
      )}
    </Layout>
  )
}
