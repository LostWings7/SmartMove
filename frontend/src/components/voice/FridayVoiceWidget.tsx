// SafeCity Loop V2 — FRIDAY Real-Time Voice Assistant Tactical HUD Widget
// Floating interactive voice companion with live audio state waveforms,
// speech recognition, quick commands, and action telemetry chips.

import React, { useState, useEffect } from 'react'
import { useVoice } from '@/context/VoiceContext'

export default function FridayVoiceWidget() {
  const {
    voiceStatus,
    isIdle,
    lastQuery,
    lastReply,
    lastActions,
    lastTools,
    sendVoiceQuery,
    toggleListening,
    stopSpeaking,
    isModalOpen,
    setIsModalOpen,
  } = useVoice()

  const [inputQuery, setInputQuery] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const [isOnline, setIsOnline] = useState(navigator.onLine)
  useEffect(() => {
    const handleOnline = () => setIsOnline(true)
    const handleOffline = () => setIsOnline(false)
    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)
    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  const isAwaitingConfirmation =
    lastActions.some((a) => a.action === 'confirm_action') ||
    lastReply.toLowerCase().includes('shall i submit it') ||
    lastReply.toLowerCase().includes('do you want me to continue') ||
    lastReply.toLowerCase().includes('sure you want to proceed') ||
    lastReply.toLowerCase().includes('shall i proceed')

  const quickPrompts = [
    'Report a pothole near the college junction',
    'Show my notifications',
    'Find me a safer route',
    'Has the repair been completed?',
    'What evidence do we have for this complaint?',
    "Show today's unresolved complaints",
    'Open my complaints',
    'Open traffic conflicts',
    'Show route comparison',
    'What are the dangerous areas?',
    'FRIDAY, sleep now',
  ]

  const handleSend = async (text: string) => {
    if (!text.trim() || isSubmitting) return
    setIsSubmitting(true)
    setInputQuery('')
    try {
      await sendVoiceQuery(text)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <>
      {/* ── Floating Voice Orb (Bottom-Right) ────────────────────────────── */}
      <div className="fixed bottom-6 right-6 z-50 flex items-center gap-3">
        {/* State Pill (Visible when speaking or listening) */}
        {(voiceStatus === 'listening' || voiceStatus === 'speaking' || voiceStatus === 'processing') && (
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-navy-900/90 border border-cyan-500/40 backdrop-blur-md shadow-lg shadow-cyan-500/10 text-xs font-mono">
            {voiceStatus === 'listening' && (
              <>
                <span className="w-2 h-2 rounded-full bg-red-500 animate-ping" />
                <span className="text-cyan-300">Listening to Boss...</span>
              </>
            )}
            {voiceStatus === 'processing' && (
              <>
                <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                <span className="text-amber-300">FRIDAY thinking...</span>
              </>
            )}
            {voiceStatus === 'speaking' && (
              <div className="flex items-center gap-1.5">
                <div className="flex items-end gap-0.5 h-3">
                  <span className="w-0.5 bg-emerald-400 animate-pulse h-2" />
                  <span className="w-0.5 bg-emerald-400 animate-pulse h-3" />
                  <span className="w-0.5 bg-emerald-400 animate-pulse h-1.5" />
                </div>
                <span className="text-emerald-300">Speaking</span>
                <button
                  onClick={stopSpeaking}
                  className="ml-1 text-gray-400 hover:text-white text-[10px] uppercase font-bold"
                >
                  Mute
                </button>
              </div>
            )}
          </div>
        )}

        {/* Main Floating Trigger Button */}
        <button
          onClick={() => setIsModalOpen(!isModalOpen)}
          className={`relative p-3.5 rounded-full shadow-2xl transition-all duration-300 flex items-center justify-center ${
            voiceStatus === 'listening'
              ? 'bg-gradient-to-r from-red-600 to-pink-600 ring-4 ring-red-400/50 scale-105 shadow-red-500/30'
              : voiceStatus === 'speaking'
              ? 'bg-gradient-to-r from-emerald-600 to-teal-600 ring-4 ring-emerald-400/50 scale-105 shadow-emerald-500/30'
              : voiceStatus === 'processing'
              ? 'bg-gradient-to-r from-amber-600 to-orange-600 ring-4 ring-amber-400/50 animate-pulse shadow-amber-500/30'
              : 'bg-gradient-to-r from-cyan-600 to-blue-700 hover:from-cyan-500 hover:to-blue-600 ring-2 ring-cyan-400/30 hover:ring-cyan-400/60 shadow-cyan-500/20'
          }`}
          title="SafeCity Voice Assistant (FRIDAY)"
          aria-label="Toggle FRIDAY Voice Assistant"
        >
          {/* Futuristic Waveform SVG / Microphone Icon */}
          <div className="text-white text-xl">
            {voiceStatus === 'listening' ? '🎙️' : voiceStatus === 'speaking' ? '🔊' : '🤖'}
          </div>
          {/* Idle Indicator Dot */}
          {isIdle && (
            <span
              className="absolute -top-1 -right-1 w-3 h-3 rounded-full bg-gray-500 border-2 border-navy-900"
              title="FRIDAY is idle (Say 'Friday' to wake)"
            />
          )}
        </button>
      </div>

      {/* ── Expanded Voice HUD Dialog ───────────────────────────────────── */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="relative w-full max-w-lg bg-navy-900 border border-cyan-500/40 rounded-2xl shadow-2xl overflow-hidden flex flex-col max-h-[85vh]">
            {/* Header */}
            <div className="px-5 py-4 border-b border-gray-800 bg-navy-800/80 flex items-center justify-between">
              <div className="flex items-center gap-2.5">
                <span className="text-xl">🤖</span>
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-white font-bold text-sm tracking-wide">FRIDAY</h2>
                    <span className="text-[10px] px-1.5 py-0.2 rounded bg-cyan-500/20 text-cyan-300 font-mono border border-cyan-500/30">
                      SafeCity Loop V2
                    </span>
                    <span
                      className={`text-[10px] px-1.5 py-0.2 rounded font-mono border flex items-center gap-1 ${
                        isOnline
                          ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
                          : 'bg-red-500/10 text-red-400 border-red-500/30'
                      }`}
                    >
                      <span className={`w-1.5 h-1.5 rounded-full ${isOnline ? 'bg-emerald-400' : 'bg-red-400'}`} />
                      {isOnline ? 'CONNECTED' : 'OFFLINE'}
                    </span>
                    {isIdle && (
                      <span className="text-[10px] px-1.5 py-0.2 rounded bg-gray-700 text-gray-300 font-mono">
                        IDLE
                      </span>
                    )}
                  </div>
                  <p className="text-gray-400 text-xs">Real-Time Tactical Voice Agent</p>
                </div>
              </div>
              <button
                onClick={() => setIsModalOpen(false)}
                className="text-gray-400 hover:text-white p-1 rounded-lg hover:bg-navy-700 transition"
              >
                ✕
              </button>
            </div>

            {/* Content & History */}
            <div className="p-5 overflow-y-auto space-y-4 flex-1">
              {/* Last User Query */}
              {lastQuery && (
                <div className="flex justify-end">
                  <div className="max-w-[85%] bg-blue-600/30 border border-blue-500/40 rounded-2xl rounded-tr-sm px-4 py-2 text-sm text-blue-100">
                    <div className="text-[10px] text-blue-300 font-semibold mb-0.5">Boss:</div>
                    {lastQuery}
                  </div>
                </div>
              )}

              {/* FRIDAY Response Card */}
              <div className="flex justify-start">
                <div className="max-w-[95%] bg-navy-800/90 border border-cyan-500/30 rounded-2xl rounded-tl-sm p-4 text-sm text-gray-100 shadow-inner">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[11px] text-cyan-400 font-mono font-semibold flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />
                      FRIDAY:
                    </span>
                    {voiceStatus === 'speaking' && (
                      <button
                        onClick={stopSpeaking}
                        className="text-[10px] text-red-400 hover:text-red-300 font-mono font-bold"
                      >
                        Stop Audio
                      </button>
                    )}
                  </div>
                  <p className="leading-relaxed text-sm text-gray-100 whitespace-pre-wrap">{lastReply}</p>

                  {/* Executed Tools Telemetry Chips */}
                  {lastTools && lastTools.length > 0 && (
                    <div className="mt-3 pt-2.5 border-t border-gray-700/60 flex flex-wrap gap-1.5">
                      {lastTools.map((t, i) => (
                        <span
                          key={i}
                          className="text-[10px] font-mono px-2 py-0.5 rounded bg-navy-950 text-cyan-300 border border-cyan-500/20"
                        >
                          🔧 {t.tool}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Executed Frontend Actions Chips */}
                  {lastActions && lastActions.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {lastActions.map((a, i) => (
                        <span
                          key={i}
                          className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-300 border border-emerald-500/30"
                        >
                          ⚡ Action: {a.action}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              </div>

              {/* Mandatory Consequential Action Confirmation Prompt (Phase 4.1 & 4.4) */}
              {isAwaitingConfirmation && (
                <div className="p-3.5 bg-amber-500/15 border border-amber-500/40 rounded-xl flex items-center justify-between gap-3 shadow-lg shadow-amber-500/10">
                  <div className="text-xs text-amber-200 flex items-center gap-2">
                    <span className="text-base">⚠️</span>
                    <div>
                      <span className="font-semibold block text-amber-100">Action Authorization Required</span>
                      <span className="text-[11px] text-amber-300/80">Speak 'Yes' / 'No' or click below to authorize.</span>
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => handleSend('Yes, confirm')}
                      disabled={isSubmitting}
                      className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-xs font-semibold rounded-lg shadow transition"
                    >
                      Authorize
                    </button>
                    <button
                      onClick={() => handleSend('No, cancel')}
                      disabled={isSubmitting}
                      className="px-3 py-1.5 bg-red-600/80 hover:bg-red-500 disabled:opacity-50 text-white text-xs font-semibold rounded-lg shadow transition"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}

              {/* Quick Suggestion Pills */}
              <div>
                <p className="text-xs text-gray-400 mb-2 font-medium">SafeCity Voice Actions:</p>
                <div className="flex flex-wrap gap-1.5">
                  {quickPrompts.map((q, i) => (
                    <button
                      key={i}
                      onClick={() => handleSend(q)}
                      disabled={isSubmitting}
                      className="text-xs px-2.5 py-1 rounded-lg bg-navy-800 hover:bg-navy-700 border border-gray-700 hover:border-cyan-500/50 text-gray-300 hover:text-cyan-300 transition text-left"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Input Footer */}
            <div className="p-4 border-t border-gray-800 bg-navy-850 flex items-center gap-2">
              <button
                onClick={toggleListening}
                className={`p-2.5 rounded-xl border transition ${
                  voiceStatus === 'listening'
                    ? 'bg-red-500/20 border-red-500/60 text-red-400 animate-pulse'
                    : 'bg-navy-800 border-gray-700 text-gray-300 hover:text-white hover:border-cyan-500/50'
                }`}
                title={voiceStatus === 'listening' ? 'Stop listening' : 'Start speaking'}
              >
                🎙️
              </button>
              <input
                type="text"
                value={inputQuery}
                onChange={(e) => setInputQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSend(inputQuery)}
                placeholder="Speak or type a command to FRIDAY..."
                className="flex-1 bg-navy-950 border border-gray-700 focus:border-cyan-500 rounded-xl px-3.5 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:ring-1 focus:ring-cyan-500 font-sans"
              />
              <button
                onClick={() => handleSend(inputQuery)}
                disabled={!inputQuery.trim() || isSubmitting}
                className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-40 text-white rounded-xl text-sm font-semibold transition"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
