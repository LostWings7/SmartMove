import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useAuth } from '@/context/AuthContext'

export default function Register() {
  const { register } = useAuth()
  const navigate = useNavigate()

  const [fullName, setFullName] = useState('')
  const [email, setEmail] = useState('')
  const [phone, setPhone] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [agreeTerms, setAgreeTerms] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isSubmitting, setIsSubmitting] = useState(false)

  // Basic password strength meter
  function getPasswordStrength(p: string) {
    if (!p) return 0
    let score = 0
    if (p.length >= 8) score += 1
    if (/[A-Z]/.test(p)) score += 1
    if (/[0-9]/.test(p)) score += 1
    if (/[^A-Za-z0-9]/.test(p)) score += 1
    return score
  }

  const strength = getPasswordStrength(password)
  const strengthLabels = ['Weak', 'Moderate', 'Good', 'Strong']
  const strengthColors = ['bg-red-500', 'bg-amber-500', 'bg-blue-500', 'bg-emerald-500']

  async function handleRegisterSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError(null)

    if (password !== confirmPassword) {
      setError('Passwords do not match.')
      return
    }

    if (password.length < 8) {
      setError('Password must be at least 8 characters in length.')
      return
    }

    if (!agreeTerms) {
      setError('You must accept the terms of civic privacy and data handling.')
      return
    }

    setIsSubmitting(true)
    try {
      await register(fullName, email, password, phone)
      navigate('/citizen/dashboard')
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Registration failed. Please check your inputs.')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="min-h-screen bg-navy-950 flex flex-col justify-center items-center px-4 py-12 relative overflow-hidden">
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-accent/5 rounded-full blur-3xl pointer-events-none" />

      {/* Header */}
      <div className="text-center mb-6 relative z-10">
        <div className="inline-flex items-center justify-center w-12 h-12 bg-accent rounded-xl text-white font-black text-xl shadow-lg shadow-accent/30 mb-3">
          SC
        </div>
        <h1 className="text-2xl font-bold text-white tracking-tight">SafeCity Citizen Portal</h1>
        <p className="text-gray-400 text-xs mt-1">Create an authoritative account to file and track municipal road complaints</p>
      </div>

      {/* Registration Card */}
      <div className="w-full max-w-md bg-navy-800/90 border border-gray-800 rounded-2xl p-6 sm:p-8 backdrop-blur-md shadow-2xl relative z-10">
        <h2 className="text-lg font-semibold text-white mb-1">Create Citizen Account</h2>
        <p className="text-gray-400 text-xs mb-5">Your details remain protected and confidential under municipal privacy rules.</p>

        {error && (
          <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-xl text-red-400 text-xs flex items-center gap-2">
            <span>⚠</span>
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleRegisterSubmit} className="space-y-3.5">
          <div>
            <label className="block text-xs font-medium text-gray-300 mb-1">Full Name</label>
            <input
              type="text"
              required
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              placeholder="e.g. Priya Sharma"
              className="w-full bg-navy-900 border border-gray-700 rounded-xl px-3.5 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent transition"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-300 mb-1">Email Address</label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="e.g. priya.sharma@example.com"
              className="w-full bg-navy-900 border border-gray-700 rounded-xl px-3.5 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent transition"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-300 mb-1">Phone Number (Optional)</label>
            <input
              type="tel"
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              placeholder="+91 98450 12345"
              className="w-full bg-navy-900 border border-gray-700 rounded-xl px-3.5 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent transition"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-300 mb-1">Password</label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="At least 8 characters"
              className="w-full bg-navy-900 border border-gray-700 rounded-xl px-3.5 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent transition"
            />
            {password && (
              <div className="mt-1.5 flex items-center gap-2">
                <div className="flex-1 h-1 bg-navy-900 rounded-full overflow-hidden flex gap-0.5">
                  {[1, 2, 3, 4].map((step) => (
                    <div
                      key={step}
                      className={`h-full flex-1 ${step <= strength ? strengthColors[strength - 1] : 'bg-gray-700'}`}
                    />
                  ))}
                </div>
                <span className="text-[10px] text-gray-400 font-mono">
                  {strengthLabels[strength - 1] || 'Too short'}
                </span>
              </div>
            )}
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-300 mb-1">Confirm Password</label>
            <input
              type="password"
              required
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="Re-enter password"
              className="w-full bg-navy-900 border border-gray-700 rounded-xl px-3.5 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-accent transition"
            />
          </div>

          <div className="flex items-start gap-2 pt-1">
            <input
              type="checkbox"
              id="terms"
              checked={agreeTerms}
              onChange={(e) => setAgreeTerms(e.target.checked)}
              className="mt-0.5 rounded border-gray-700 bg-navy-900 text-accent focus:ring-accent"
            />
            <label htmlFor="terms" className="text-[11px] text-gray-400 leading-tight">
              I agree to the SafeCity public complaint guidelines and consent to photographic evidence verification.
            </label>
          </div>

          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full mt-2 py-3 bg-accent hover:bg-accent/90 disabled:opacity-50 text-white font-medium text-sm rounded-xl transition shadow-lg shadow-accent/20 flex items-center justify-center gap-2"
          >
            {isSubmitting ? (
              <>
                <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Creating Account...</span>
              </>
            ) : (
              'Complete Registration'
            )}
          </button>
        </form>

        <div className="mt-5 text-center text-xs text-gray-400">
          Already registered?{' '}
          <Link to="/login" className="text-accent hover:underline font-medium">
            Sign In
          </Link>
        </div>
      </div>
    </div>
  )
}
