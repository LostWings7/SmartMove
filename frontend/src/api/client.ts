import axios from 'axios'

// API base URL — reads from Vite env or defaults to same origin via proxy
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

export const api = axios.create({
  baseURL: BASE_URL,
  timeout: 10_000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor attaches Bearer token if present
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('safecity_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Response interceptor for global error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401 && !window.location.pathname.startsWith('/login') && !window.location.pathname.startsWith('/register')) {
      // Clear expired token
      localStorage.removeItem('safecity_token')
      localStorage.removeItem('safecity_user')
    }
    if (error.response?.status === 500) {
      console.error('[SafeCity API] Server error:', error.response.data)
    }
    return Promise.reject(error)
  },
)
