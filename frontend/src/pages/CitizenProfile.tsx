import { useState } from 'react'
import Layout from '@/components/layout/Layout'
import { useAuth } from '@/context/AuthContext'
import { api } from '@/api/client'

export default function CitizenProfile() {
  const { user, updateUser } = useAuth()

  const [fullName, setFullName] = useState(user?.full_name || '')
  const [phone, setPhone] = useState(user?.phone_number || '')
  const [preferredLang, setPreferredLang] = useState(user?.preferred_language || 'en')
  const [emailNotif, setEmailNotif] = useState(user?.notification_preferences?.email ?? true)
  const [inAppNotif, setInAppNotif] = useState(user?.notification_preferences?.in_app ?? true)

  const [savingProfile, setSavingProfile] = useState(false)
  const [profileSuccess, setProfileSuccess] = useState<string | null>(null)
  const [profileError, setProfileError] = useState<string | null>(null)

  // Password change state
  const [currentPassword, setCurrentPassword] = useState('')
  const [newPassword, setNewPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [savingPassword, setSavingPassword] = useState(false)
  const [passwordSuccess, setPasswordSuccess] = useState<string | null>(null)
  const [passwordError, setPasswordError] = useState<string | null>(null)

  async function handleProfileSave(e: React.FormEvent) {
    e.preventDefault()
    setSavingProfile(true)
    setProfileSuccess(null)
    setProfileError(null)
    try {
      const res = await api.patch('/api/auth/profile', {
        full_name: fullName.trim(),
        phone_number: phone.trim() || null,
        preferred_language: preferredLang,
        notification_preferences: {
          email: emailNotif,
          in_app: inAppNotif,
        },
      })
      updateUser(res.data)
      setProfileSuccess('Profile details saved successfully.')
    } catch (err: any) {
      setProfileError(err.response?.data?.detail || 'Failed to update profile.')
    } finally {
      setSavingProfile(false)
    }
  }

  async function handlePasswordChange(e: React.FormEvent) {
    e.preventDefault()
    if (newPassword !== confirmPassword) {
      setPasswordError('New passwords do not match.')
      return
    }
    if (newPassword.length < 8) {
      setPasswordError('New password must be at least 8 characters.')
      return
    }

    setSavingPassword(true)
    setPasswordSuccess(null)
    setPasswordError(null)
    try {
      await api.post('/api/auth/change-password', {
        current_password: currentPassword,
        new_password: newPassword,
      })
      setPasswordSuccess('Password successfully updated.')
      setCurrentPassword('')
      setNewPassword('')
      setConfirmPassword('')
    } catch (err: any) {
      setPasswordError(err.response?.data?.detail || 'Failed to change password.')
    } finally {
      setSavingPassword(false)
    }
  }

  return (
    <Layout
      title="Citizen Profile & Settings"
      subtitle="Manage your contact details, notification preferences, and account security"
    >
      <div className="max-w-3xl mx-auto space-y-8">
        {/* Profile Card */}
        <div className="bg-navy-800 border border-gray-800 rounded-2xl p-6 sm:p-8 space-y-6">
          <div className="border-b border-gray-800 pb-4">
            <h3 className="text-base font-semibold text-white">Personal Information</h3>
            <p className="text-gray-400 text-xs mt-0.5">Authoritative citizen identification details.</p>
          </div>

          {profileSuccess && (
            <div className="p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-emerald-400 text-xs">
              ✓ {profileSuccess}
            </div>
          )}
          {profileError && (
            <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-xs">
              ⚠ {profileError}
            </div>
          )}

          <form onSubmit={handleProfileSave} className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1">Full Name</label>
                <input
                  type="text"
                  required
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  className="w-full bg-navy-900 border border-gray-700 rounded-xl px-3.5 py-2.5 text-xs text-white focus:outline-none focus:border-accent"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1">
                  Email Address <span className="text-gray-500">(Authoritative ID)</span>
                </label>
                <input
                  type="email"
                  disabled
                  value={user?.email || ''}
                  className="w-full bg-navy-900/50 border border-gray-800 rounded-xl px-3.5 py-2.5 text-xs text-gray-400 cursor-not-allowed font-mono"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1">Phone Number</label>
                <input
                  type="tel"
                  value={phone}
                  onChange={(e) => setPhone(e.target.value)}
                  placeholder="+91 98450 12345"
                  className="w-full bg-navy-900 border border-gray-700 rounded-xl px-3.5 py-2.5 text-xs text-white focus:outline-none focus:border-accent"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1">System Role</label>
                <div className="bg-navy-900/50 border border-gray-800 rounded-xl px-3.5 py-2.5 text-xs text-emerald-400 font-semibold font-mono flex items-center justify-between">
                  <span>{user?.role}</span>
                  <span className="text-[10px] text-gray-500">Read-Only</span>
                </div>
              </div>
            </div>

            {/* Notification Preferences */}
            <div className="pt-2 border-t border-gray-800 space-y-3">
              <label className="block text-xs font-semibold text-gray-300">Notification Preferences</label>
              <div className="flex flex-col sm:flex-row gap-4">
                <label className="flex items-center gap-2 cursor-pointer text-xs text-gray-300">
                  <input
                    type="checkbox"
                    checked={inAppNotif}
                    onChange={(e) => setInAppNotif(e.target.checked)}
                    className="rounded border-gray-700 bg-navy-900 text-accent focus:ring-accent"
                  />
                  <span>Receive In-App Status Alerts</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer text-xs text-gray-300">
                  <input
                    type="checkbox"
                    checked={emailNotif}
                    onChange={(e) => setEmailNotif(e.target.checked)}
                    className="rounded border-gray-700 bg-navy-900 text-accent focus:ring-accent"
                  />
                  <span>Receive Municipal Resolution Emails</span>
                </label>
              </div>
            </div>

            <div className="flex justify-end pt-3">
              <button
                type="submit"
                disabled={savingProfile}
                className="px-6 py-2.5 bg-accent hover:bg-accent/90 disabled:opacity-50 text-white font-medium text-xs rounded-xl transition"
              >
                {savingProfile ? 'Saving...' : 'Save Profile Changes'}
              </button>
            </div>
          </form>
        </div>

        {/* Password Security Card */}
        <div className="bg-navy-800 border border-gray-800 rounded-2xl p-6 sm:p-8 space-y-6">
          <div className="border-b border-gray-800 pb-4">
            <h3 className="text-base font-semibold text-white">Account Security & Password</h3>
            <p className="text-gray-400 text-xs mt-0.5">Change your password by verifying your current secret.</p>
          </div>

          {passwordSuccess && (
            <div className="p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-emerald-400 text-xs">
              ✓ {passwordSuccess}
            </div>
          )}
          {passwordError && (
            <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-xs">
              ⚠ {passwordError}
            </div>
          )}

          <form onSubmit={handlePasswordChange} className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1">Current Password</label>
              <input
                type="password"
                required
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                className="w-full sm:max-w-md bg-navy-900 border border-gray-700 rounded-xl px-3.5 py-2.5 text-xs text-white focus:outline-none focus:border-accent"
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1">New Password (Min 8 chars)</label>
                <input
                  type="password"
                  required
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  className="w-full bg-navy-900 border border-gray-700 rounded-xl px-3.5 py-2.5 text-xs text-white focus:outline-none focus:border-accent"
                />
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-300 mb-1">Confirm New Password</label>
                <input
                  type="password"
                  required
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full bg-navy-900 border border-gray-700 rounded-xl px-3.5 py-2.5 text-xs text-white focus:outline-none focus:border-accent"
                />
              </div>
            </div>

            <div className="flex justify-end pt-3">
              <button
                type="submit"
                disabled={savingPassword}
                className="px-6 py-2.5 bg-navy-700 hover:bg-navy-600 disabled:opacity-50 text-white font-medium text-xs rounded-xl border border-gray-700 transition"
              >
                {savingPassword ? 'Updating Password...' : 'Update Password'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </Layout>
  )
}
