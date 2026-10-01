// SafeCity Loop V2 — Voice Context & Real-Time Assistant Provider
// Connects voice state, query processing, action dispatching, and Web Speech audio.

import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { api } from '@/api/client'
import { useAuth } from '@/context/AuthContext'
import { voiceActionDispatcher } from '@/services/voiceActionDispatcher'
import type { VoiceAction, VoiceQueryResult, SafeCityContextState, VoiceToolExecution } from '@/types/voice'

interface VoiceContextType {
  voiceStatus: 'idle' | 'listening' | 'speaking' | 'processing' | 'disabled'
  isVoiceActive: boolean
  isIdle: boolean
  lastQuery: string
  lastReply: string
  lastActions: VoiceAction[]
  lastTools: VoiceToolExecution[]
  isModalOpen: boolean
  setIsModalOpen: (open: boolean) => void
  sendVoiceQuery: (query: string, overrideContext?: Record<string, any>) => Promise<VoiceQueryResult>
  toggleListening: () => void
  speakText: (text: string, onEnd?: () => void) => void
  speakNavigationClue: (clueText: string, onEnd?: () => void) => void
  stopSpeaking: () => void
  syncContext: (updates: Partial<SafeCityContextState>) => Promise<void>
}

const VoiceContext = createContext<VoiceContextType | undefined>(undefined)

export function VoiceProvider({ children }: { children: React.ReactNode }) {
  const { user, isAuthenticated } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()

  const [voiceStatus, setVoiceStatus] = useState<'idle' | 'listening' | 'speaking' | 'processing' | 'disabled'>('idle')
  const [isIdle, setIsIdle] = useState<boolean>(false)
  const [lastQuery, setLastQuery] = useState<string>('')
  const [lastReply, setLastReply] = useState<string>(
    'FRIDAY online. Standing by for SafeCity operations, Boss.'
  )
  const [lastActions, setLastActions] = useState<VoiceAction[]>([])
  const [lastTools, setLastTools] = useState<VoiceToolExecution[]>([])
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false)

  const recognitionRef = useRef<any>(null)
  const sessionIdRef = useRef<string>(`session_${Date.now()}`)

  // Track page path changes and synchronize context with backend
  useEffect(() => {
    const pagePath = location.pathname
    api.post('/api/voice/context', {
      session_id: sessionIdRef.current,
      current_page: pagePath,
    }).catch(() => {})
  }, [location.pathname])

  // Synchronize authenticated user identity
  useEffect(() => {
    if (user && isAuthenticated) {
      api.post('/api/voice/context', {
        session_id: sessionIdRef.current,
        current_user: {
          id: user.id,
          full_name: user.full_name,
          email: user.email,
          role: user.role,
        },
        user_role: user.role,
      }).catch(() => {})
    }
  }, [user, isAuthenticated])

  const stopSpeaking = useCallback(() => {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel()
    }
    setVoiceStatus('idle')
  }, [])

  const speakText = useCallback((text: string, onEnd?: () => void) => {
    if (!('speechSynthesis' in window)) {
      if (onEnd) onEnd()
      return
    }
    window.speechSynthesis.cancel()

    // Clean text of technical markup for clean speech
    const cleanSpeech = text
      .replace(/[*_#`~]/g, '')
      .replace(/https?:\/\/\S+/g, '')
      .trim()

    if (!cleanSpeech) {
      if (onEnd) onEnd()
      return
    }

    const utterance = new SpeechSynthesisUtterance(cleanSpeech)
    utterance.rate = 1.05
    utterance.pitch = 1.0

    // Try finding an English female/clear voice if possible (similar to Aoede)
    const voices = window.speechSynthesis.getVoices()
    const preferredVoice = voices.find(
      (v) => v.lang.startsWith('en') && (v.name.includes('Female') || v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha') || v.name.includes('Zira'))
    ) || voices.find((v) => v.lang.startsWith('en'))
    if (preferredVoice) {
      utterance.voice = preferredVoice
    }

    utterance.onstart = () => setVoiceStatus('speaking')
    utterance.onend = () => {
      setVoiceStatus('idle')
      if (onEnd) onEnd()
    }
    utterance.onerror = () => {
      setVoiceStatus('idle')
      if (onEnd) onEnd()
    }

    window.speechSynthesis.speak(utterance)
  }, [])

  const speakNavigationClue = useCallback((clueText: string, onEnd?: () => void) => {
    // Verbatim recitation of displayed turn-by-turn navigation clues
    speakText(clueText, onEnd)
  }, [speakText])

  const sendVoiceQuery = useCallback(
    async (queryText: string, overrideContext?: Record<string, any>): Promise<VoiceQueryResult> => {
      const clean = queryText.trim()
      if (!clean) {
        return {
          success: false,
          reply: 'No query provided, Boss.',
          emotion: 'neutral',
          is_idle: false,
          tools_called: [],
          frontend_actions: [],
        }
      }

      setLastQuery(clean)
      setVoiceStatus('processing')

      try {
        const payload: Record<string, any> = {
          session_id: sessionIdRef.current,
          query: clean,
        }
        if (overrideContext) {
          payload.context_override = overrideContext
        }

        const res = await api.post<VoiceQueryResult>('/api/voice/query', payload)
        const data = res.data

        setLastReply(data.reply)
        setIsIdle(data.is_idle)
        setLastActions(data.frontend_actions || [])
        setLastTools(data.tools_called || [])

        // Execute allowlisted frontend actions returned by FRIDAY
        if (data.frontend_actions && data.frontend_actions.length > 0) {
          for (const action of data.frontend_actions) {
            await voiceActionDispatcher.dispatch(action, navigate, user?.role || 'CITIZEN')
          }
        }

        // Verbalize reply using Web Speech API
        speakText(data.reply)

        return data
      } catch (err: any) {
        const errMsg = err.response?.data?.detail || 'Voice communication failed. SafeCity edge pipeline offline.'
        setLastReply(errMsg)
        setVoiceStatus('idle')
        return {
          success: false,
          reply: errMsg,
          emotion: 'neutral',
          is_idle: false,
          tools_called: [],
          frontend_actions: [],
          error: errMsg,
        }
      }
    },
    [navigate, user, speakText]
  )

  const syncContext = useCallback(async (updates: Partial<SafeCityContextState>) => {
    try {
      await api.post('/api/voice/context', {
        session_id: sessionIdRef.current,
        ...updates,
      })
    } catch {
      // Non-blocking sync failure
    }
  }, [])

  // Web Speech API Microphone setup
  const toggleListening = useCallback(() => {
    if (voiceStatus === 'listening') {
      if (recognitionRef.current) {
        recognitionRef.current.stop()
      }
      setVoiceStatus('idle')
      return
    }

    // Stop speaking if currently talking
    stopSpeaking()

    const SpeechRecognition =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition

    if (!SpeechRecognition) {
      // Speech recognition not supported by browser; prompt for text
      setIsModalOpen(true)
      return
    }

    try {
      const recognition = new SpeechRecognition()
      recognition.continuous = false
      recognition.interimResults = false
      recognition.lang = 'en-US'

      recognition.onstart = () => {
        setVoiceStatus('listening')
      }

      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript
        setVoiceStatus('processing')
        sendVoiceQuery(transcript)
      }

      recognition.onerror = () => {
        setVoiceStatus('idle')
      }

      recognition.onend = () => {
        setVoiceStatus((prev) => (prev === 'listening' ? 'idle' : prev))
      }

      recognitionRef.current = recognition
      recognition.start()
    } catch {
      setVoiceStatus('idle')
      setIsModalOpen(true)
    }
  }, [voiceStatus, sendVoiceQuery, stopSpeaking])

  return (
    <VoiceContext.Provider
      value={{
        voiceStatus,
        isVoiceActive: voiceStatus !== 'disabled',
        isIdle,
        lastQuery,
        lastReply,
        lastActions,
        lastTools,
        isModalOpen,
        setIsModalOpen,
        sendVoiceQuery,
        toggleListening,
        speakText,
        speakNavigationClue,
        stopSpeaking,
        syncContext,
      }}
    >
      {children}
    </VoiceContext.Provider>
  )
}

export function useVoice(): VoiceContextType {
  const context = useContext(VoiceContext)
  if (!context) {
    throw new Error('useVoice must be used within a VoiceProvider')
  }
  return context
}
