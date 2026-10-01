import { Navigate, useLocation } from 'react-router-dom'
import { useAuth } from '@/context/AuthContext'
import type { UserRole } from '@/types'

interface ProtectedRouteProps {
  children: React.ReactNode
  allowedRoles?: UserRole[]
}

export default function ProtectedRoute({ children, allowedRoles }: ProtectedRouteProps) {
  const { user, isAuthenticated, isLoading } = useAuth()
  const location = useLocation()

  if (isLoading) {
    return (
      <div className="min-h-screen bg-navy-950 flex flex-col items-center justify-center text-white">
        <div className="w-10 h-10 border-2 border-accent border-t-transparent rounded-full animate-spin mb-4" />
        <p className="text-gray-400 text-sm font-mono tracking-wide">Authenticating SafeCity credentials...</p>
      </div>
    )
  }

  if (!isAuthenticated || !user) {
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  if (allowedRoles && !allowedRoles.includes(user.role)) {
    // If citizen tries to access admin routes, redirect to citizen dashboard
    if (user.role === 'CITIZEN') {
      return <Navigate to="/citizen/dashboard" replace />
    }
    // If non-admin tries to access superuser routes, redirect to overview
    return <Navigate to="/" replace />
  }

  return <>{children}</>
}
