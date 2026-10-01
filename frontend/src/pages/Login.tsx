import { useState } from 'react'
import { Link, useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '@/context/AuthContext'

export default function Login() {
  const { login } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const from = (location.state as any)?.from?.pathname || '/'

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  const demoAccounts = [
    { label: 'Citizen', name: 'Priya Sharma', email: 'citizen_demo@safecity.gov', pass: 'CitizenPass123!', role: 'CITIZEN', color: 'emerald' },
    { label: 'Inspector', name: 'Marcus Chen', email: 'inspector_demo@safecity.gov', pass: 'InspectorPass123!', role: 'INSPECTOR', color: 'cyan' },
    { label: 'Engineer', name: 'Elena Rostova', email: 'engineer_demo@safecity.gov', pass: 'EngineerPass123!', role: 'ENGINEER', color: 'indigo' },
    { label: 'Admin', name: 'David Vance', email: 'admin_demo@safecity.gov', pass: 'AdminPass123!', role: 'ADMIN', color: 'amber' },
    { label: 'Superuser', name: 'Comm. Rao', email: 'superuser_demo@safecity.gov', pass: 'SuperPass123!', role: 'SUPERUSER', color: 'purple' },
  ]

  async function handleLoginSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)
    setIsSubmitting(true)
    try {
      const loggedUser = await login(email, password)
      if (loggedUser.role === 'CITIZEN') {
        navigate('/citizen/dashboard')
      } else {
        navigate(from === '/' ? '/admin/complaints' : from)
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Login failed. Please check your credentials.')
    } finally {
      setIsSubmitting(false)
    }
  }

  async function fillAndLoginDemo(demoEmail: string, demoPass: string, role: string) {
    setEmail(demoEmail)
    setPassword(demoPass)
    setError(null)
    setIsSubmitting(true)
    try {
      const loggedUser = await login(demoEmail, demoPass)
      if (loggedUser.role === 'CITIZEN') {
        navigate('/citizen/dashboard')
      } else {
        navigate('/admin/complaints')
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Demo login failed.')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen bg-navy-950 flex flex-col justify-center items-center px-4 py-12 relative overflow-hidden">
      {/* Background visual elements */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-accent/5 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute bottom-10 left-10 w-72 h-72 bg-blue-600/5 rounded-full blur-2xl pointer-events-none" />

      {/* Header */}
      <div className="text-center mb-8 relative z-10">
        <div className="inline-flex items-center justify-center w-12 h-12 bg-accent rounded-xl text-white font-black text-xl shadow-lg shadow-accent/30 mb-3">
          SC
        </div>
        <h1 className="text-2xl font-bold text-white tracking-tight">SafeCity Loop V2</h1>
        <p className="text-gray-400 text-xs mt-1">Authoritative Municipal Road Intelligence & Citizen Portal</p>
      </div>

      {/* Main Login Card */}
      <div className="w-full max-w-md bg-navy-800/90 border border-gray-800 rounded-2xl p-6 sm:p-8 backdrop-blur-md shadow-2xl relative z-10">
        <h2 className="text-lg font-semibold text-white mb-1">Sign In to Your Account</h2>
        <p className="text-gray-400 text-xs mb-6">Enter your registered email and password to access the portal.</p>

        {error && (
          <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-xs flex items-center gap-2">
            <span>⚠</span>
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleLoginSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-gray-300 mb-1.5">Email Address</label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="e.g. citizen_demo@safecity.gov"
              className="w-full bg-navy-900 border border-gray-700 rounded-xl px-3.5 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent transition"
            />
          </div>

          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-medium text-gray-300">Password</label>
            </div>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••••••"
              className="w-full bg-navy-900 border border-gray-700 rounded-xl px-3.5 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent transition"
            />
          </div>

          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full mt-2 py-3 bg-accent hover:bg-accent/90 disabled:opacity-50 text-white font-medium text-sm rounded-xl transition shadow-lg shadow-accent/20 flex items-center justify-center gap-2"
          >
            {isSubmitting ? (
              <>
                <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Signing In...</span>
              </>
            ) : (
              'Sign In'
            )}
          </button>
        </form>

        <div className="mt-5 text-center text-xs text-gray-400">
          Don't have an account yet?{' '}
          <Link to="/register" className="text-accent hover:underline font-medium">
            Register as Citizen
          </Link>
        </div>

        {/* 1-Click Demo Logins */}
        <div className="mt-8 pt-6 border-t border-gray-800">
          <div className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-2.5 text-center">
            ⚡ Quick Demo Logins (1-Click)
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
            {demoAccounts.map((acc) => (
              <button
                key={acc.role}
                type="button"
                onClick={() => fillAndLoginDemo(acc.email, acc.pass, acc.role)}
                className="text-left p-2 rounded-lg bg-navy-900/80 hover:bg-navy-700 border border-gray-800 hover:border-gray-700 transition"
              >
                <div className="text-[11px] font-semibold text-white truncate">{acc.label}</div>
                <div className="text-[9px] text-gray-400 truncate">{acc.name}</div>
              </button>
            ))}
          </div>
          <p className="text-[10px] text-gray-500 text-center mt-2.5">
            Passwords pre-seeded deterministically for rapid evaluation.
          </p>
        </div>
      </div>
    </div>
  )
}
