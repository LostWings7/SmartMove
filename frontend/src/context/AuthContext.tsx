import React, { createContext, useContext, useState, useEffect } from 'react'
import { api } from '@/api/client'
import type { User, AuthResponse } from '@/types'

interface AuthContextType {
  user: User | null
  token: string | null
  isAuthenticated: boolean
  isLoading: boolean
  login: (email: string, password: string) => Promise<User>
  register: (fullName: string, email: string, password: string, phone?: string) => Promise<User>
  logout: () => void
  updateUser: (user: User) => void
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(() => {
    try {
      const stored = localStorage.getItem('safecity_user')
      return stored ? JSON.parse(stored) : null
    } catch {
      return null
    }
  })
  const [token, setToken] = useState<string | null>(() => localStorage.getItem('safecity_token'))
  const [isLoading, setIsLoading] = useState<boolean>(true)

  useEffect(() => {
    async function verifyAuth() {
      const storedToken = localStorage.getItem('safecity_token')
      if (storedToken) {
        try {
          const res = await api.get<User>('/api/auth/me')
          setUser(res.data)
          localStorage.setItem('safecity_user', JSON.stringify(res.data))
        } catch {
          // Token expired or invalid
          localStorage.removeItem('safecity_token')
          localStorage.removeItem('safecity_user')
          setUser(null)
          setToken(null)
        }
      }
      setIsLoading(false)
    }
    verifyAuth()
  }, [])

  async function login(email: string, password: string): Promise<User> {
    const res = await api.post<AuthResponse>('/api/auth/login', { email, password })
    const { access_token, user: loggedInUser } = res.data
    localStorage.setItem('safecity_token', access_token)
    localStorage.setItem('safecity_user', JSON.stringify(loggedInUser))
    setToken(access_token)
    setUser(loggedInUser)
    return loggedInUser
  }

  async function register(fullName: string, email: string, password: string, phone?: string): Promise<User> {
    const res = await api.post<AuthResponse>('/api/auth/register', {
      full_name: fullName,
      email,
      password,
      phone_number: phone,
    })
    const { access_token, user: registeredUser } = res.data
    localStorage.setItem('safecity_token', access_token)
    localStorage.setItem('safecity_user', JSON.stringify(registeredUser))
    setToken(access_token)
    setUser(registeredUser)
    return registeredUser
  }

  function logout() {
    try {
      api.post('/api/auth/logout').catch(() => {})
    } finally {
      localStorage.removeItem('safecity_token')
      localStorage.removeItem('safecity_user')
      setUser(null)
      setToken(null)
      window.location.href = '/login'
    }
  }

  function updateUser(updatedUser: User) {
    setUser(updatedUser)
    localStorage.setItem('safecity_user', JSON.stringify(updatedUser))
  }

  return (
    <AuthContext.Provider
      value={{
        user,
        token,
        isAuthenticated: !!user && !!token,
        isLoading,
        login,
        register,
        logout,
        updateUser,
      }}
    >
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
