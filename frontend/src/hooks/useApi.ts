import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'
import type {
  DashboardData, Road, Hazard, Junction, Intervention, CitizenReport, CitizenReportCreate,
  PotholeAnalysisResult, TrafficAnalysisResult, AIModelStatus, AIEvent,
  RoadSafetyEvaluation, PrioritizedRepairItem, MunicipalRepairStatus,
  RouteComparisonResult, SmartJunctionDisplayState,
  RoadSegmentDossier, EvidenceFile, AuditLog, NearMiss, DecisionAuditRecord,
  RoadConditionAnalysisResult,
} from '@/types'

// ── Dashboard ─────────────────────────────────────────────────────────────────
export function useDashboard() {
  return useQuery<DashboardData>({
    queryKey: ['dashboard'],
    queryFn: async () => (await api.get('/api/dashboard')).data,
  })
}

// ── Roads ─────────────────────────────────────────────────────────────────────
export function useRoads(params?: { status?: string; min_risk?: number }) {
  return useQuery<Road[]>({
    queryKey: ['roads', params],
    queryFn: async () => (await api.get('/api/roads', { params })).data,
  })
}

// ── Hazards ───────────────────────────────────────────────────────────────────
export function useHazards(params?: { type?: string; severity?: string; road_id?: number }) {
  return useQuery<Hazard[]>({
    queryKey: ['hazards', params],
    queryFn: async () => (await api.get('/api/hazards', { params })).data,
  })
}

// ── Danger Zones / Safety Evaluations ────────────────────────────────────────
export function useDangerZones() {
  return useQuery<RoadSafetyEvaluation[]>({
    queryKey: ['danger-zones'],
    queryFn: async () => (await api.get('/api/danger-zones')).data,
  })
}

export function useDangerZone(roadId: number | null) {
  return useQuery<RoadSafetyEvaluation>({
    queryKey: ['danger-zones', roadId],
    queryFn: async () => (await api.get(`/api/danger-zones/${roadId}`)).data,
    enabled: roadId !== null,
  })
}

// ── Repair Priority Queue ─────────────────────────────────────────────────────
export function useRepairPriority() {
  return useQuery<PrioritizedRepairItem[]>({
    queryKey: ['repair-priority'],
    queryFn: async () => (await api.get('/api/repair-priority')).data,
  })
}

export function useUpdateRepairStatus() {
  const queryClient = useQueryClient()
  return useMutation<
    PrioritizedRepairItem,
    Error,
    { id: number; status?: MunicipalRepairStatus; action?: string; assigned_to?: string; notes?: string; reviewer?: string }
  >({
    mutationFn: async ({ id, status, action, assigned_to, notes, reviewer }) =>
      (await api.patch(`/api/repairs/${id}`, { status: status || 'NEW', action, assigned_to, notes, reviewer })).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['repair-priority'] })
      queryClient.invalidateQueries({ queryKey: ['dashboard'] })
      queryClient.invalidateQueries({ queryKey: ['danger-zones'] })
      queryClient.invalidateQueries({ queryKey: ['roads'] })
      queryClient.invalidateQueries({ queryKey: ['interventions'] })
      queryClient.invalidateQueries({ queryKey: ['audit-logs'] })
      queryClient.invalidateQueries({ queryKey: ['decision-audits'] })
    },
  })
}

// ── Junctions ─────────────────────────────────────────────────────────────────
export function useJunctions() {
  return useQuery<Junction[]>({
    queryKey: ['junctions'],
    queryFn: async () => (await api.get('/api/junctions')).data,
  })
}

export function useJunctionDisplay(junctionId: number | null) {
  return useQuery<SmartJunctionDisplayState>({
    queryKey: ['junction-display', junctionId],
    queryFn: async () => (await api.get(`/api/junctions/${junctionId}/display`)).data,
    enabled: junctionId !== null,
    refetchInterval: 5000,
  })
}

export function useSimulateJunctionEvent() {
  const queryClient = useQueryClient()
  return useMutation<SmartJunctionDisplayState, Error, { junctionId: number; conflict_type?: string; reset?: boolean }>({
    mutationFn: async ({ junctionId, conflict_type, reset }) =>
      (await api.post(`/api/junctions/${junctionId}/simulate-event`, { conflict_type, reset })).data,
    onSuccess: (_data, variables) => {
      queryClient.invalidateQueries({ queryKey: ['junction-display', variables.junctionId] })
      queryClient.invalidateQueries({ queryKey: ['junctions'] })
      queryClient.invalidateQueries({ queryKey: ['dashboard'] })
    },
  })
}

// ── Routes ────────────────────────────────────────────────────────────────────
export function useRouteComparison(originId: number | null, destId: number | null) {
  return useQuery<RouteComparisonResult>({
    queryKey: ['routes', originId, destId],
    queryFn: async () =>
      (await api.get('/api/routes/compare', { params: { origin_id: originId, destination_id: destId } })).data,
    enabled: originId !== null && destId !== null && originId !== destId,
  })
}

// ── Interventions ─────────────────────────────────────────────────────────────
export function useInterventions() {
  return useQuery<Intervention[]>({
    queryKey: ['interventions'],
    queryFn: async () => (await api.get('/api/interventions')).data,
  })
}

// ── Citizen Reports ───────────────────────────────────────────────────────────
export function useCitizenReports() {
  return useQuery<CitizenReport[]>({
    queryKey: ['citizen-reports'],
    queryFn: async () => (await api.get('/api/reports')).data,
  })
}

export function useSubmitReport() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (data: CitizenReportCreate) =>
      api.post<CitizenReport>('/api/reports', data).then((r) => r.data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['citizen-reports'] })
      queryClient.invalidateQueries({ queryKey: ['hazards'] })
      queryClient.invalidateQueries({ queryKey: ['danger-zones'] })
      queryClient.invalidateQueries({ queryKey: ['audit-logs'] })
    },
  })
}

export function useUpdateReportStatus() {
  const queryClient = useQueryClient()
  return useMutation<CitizenReport, Error, { id: number; status: string; notes?: string }>({
    mutationFn: async ({ id, status, notes }) =>
      (await api.patch(`/api/reports/${id}/status`, { status, notes })).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['citizen-reports'] })
      queryClient.invalidateQueries({ queryKey: ['hazards'] })
      queryClient.invalidateQueries({ queryKey: ['danger-zones'] })
      queryClient.invalidateQueries({ queryKey: ['roads'] })
      queryClient.invalidateQueries({ queryKey: ['audit-logs'] })
      queryClient.invalidateQueries({ queryKey: ['decision-audits'] })
    },
  })
}

// ── AI Vision API Hooks ───────────────────────────────────────────────────────

export function useAIModelStatus() {
  return useQuery<AIModelStatus>({
    queryKey: ['ai-status'],
    queryFn: async () => (await api.get('/api/ai/status')).data,
  })
}

export function useAIEvents() {
  return useQuery<AIEvent[]>({
    queryKey: ['ai-events'],
    queryFn: async () => (await api.get('/api/ai/events')).data,
  })
}

export function useAnalyzePothole() {
  const queryClient = useQueryClient()
  return useMutation<PotholeAnalysisResult, Error, { file: File; road_id?: number; latitude?: number; longitude?: number; is_demo_sample?: boolean }>({
    mutationFn: async ({ file, road_id, latitude, longitude, is_demo_sample }) => {
      const formData = new FormData()
      formData.append('file', file)
      if (road_id !== undefined && road_id !== null) {
        formData.append('road_id', road_id.toString())
      }
      if (latitude !== undefined && latitude !== null) {
        formData.append('latitude', latitude.toString())
      }
      if (longitude !== undefined && longitude !== null) {
        formData.append('longitude', longitude.toString())
      }
      if (is_demo_sample !== undefined) {
        formData.append('is_demo_sample', is_demo_sample.toString())
      }
      const response = await api.post('/api/potholes/analyze', formData, {
        headers: { 'Content-Type': undefined },
        timeout: 120_000,
      })
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['dashboard'] })
      queryKeyRefetch(queryClient, 'hazards')
      queryKeyRefetch(queryClient, 'roads')
      queryKeyRefetch(queryClient, 'ai-events')
      queryKeyRefetch(queryClient, 'danger-zones')
      queryKeyRefetch(queryClient, 'road-intelligence')
      queryKeyRefetch(queryClient, 'evidence')
      queryKeyRefetch(queryClient, 'audit-logs')
    },
  })
}

export function useAnalyzeTraffic() {
  const queryClient = useQueryClient()
  return useMutation<TrafficAnalysisResult, Error, { file: File; junction_id?: number; road_id?: number; ttc_threshold?: number; apply_privacy?: boolean; is_demo_video?: boolean }>({
    mutationFn: async ({ file, junction_id, road_id, ttc_threshold, apply_privacy = true, is_demo_video = false }) => {
      const formData = new FormData()
      formData.append('file', file)
      if (junction_id !== undefined && junction_id !== null) {
        formData.append('junction_id', junction_id.toString())
      }
      if (road_id !== undefined && road_id !== null) {
        formData.append('road_id', road_id.toString())
      }
      if (ttc_threshold !== undefined && ttc_threshold !== null) {
        formData.append('ttc_threshold', ttc_threshold.toString())
      }
      formData.append('apply_privacy', apply_privacy.toString())
      formData.append('is_demo_video', is_demo_video.toString())
      const response = await api.post('/api/traffic/analyze', formData, {
        headers: { 'Content-Type': undefined },
        timeout: 300_000,
      })
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['dashboard'] })
      queryKeyRefetch(queryClient, 'junctions')
      queryKeyRefetch(queryClient, 'conflicts')
      queryKeyRefetch(queryClient, 'conflict-hotspots')
      queryKeyRefetch(queryClient, 'ai-events')
      queryKeyRefetch(queryClient, 'danger-zones')
      queryKeyRefetch(queryClient, 'road-intelligence')
      queryKeyRefetch(queryClient, 'evidence')
      queryKeyRefetch(queryClient, 'audit-logs')
    },
  })
}

export function useAnalyzeRoadCondition() {
  const queryClient = useQueryClient()
  return useMutation<RoadConditionAnalysisResult, Error, { file: File; road_id?: number; latitude?: number; longitude?: number }>({
    mutationFn: async ({ file, road_id, latitude, longitude }) => {
      const formData = new FormData()
      formData.append('file', file)
      if (road_id !== undefined && road_id !== null) {
        formData.append('road_id', road_id.toString())
      }
      if (latitude !== undefined && latitude !== null) {
        formData.append('latitude', latitude.toString())
      }
      if (longitude !== undefined && longitude !== null) {
        formData.append('longitude', longitude.toString())
      }
      const response = await api.post('/api/road-condition/analyze', formData, {
        headers: { 'Content-Type': undefined },
        timeout: 120_000,
      })
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['dashboard'] })
      queryKeyRefetch(queryClient, 'hazards')
      queryKeyRefetch(queryClient, 'roads')
      queryKeyRefetch(queryClient, 'ai-events')
      queryKeyRefetch(queryClient, 'danger-zones')
      queryKeyRefetch(queryClient, 'road-intelligence')
      queryKeyRefetch(queryClient, 'evidence')
      queryKeyRefetch(queryClient, 'audit-logs')
    },
  })
}

export function useDemoTrafficVideo() {
  return useQuery<{
    success: boolean
    video_url: string
    filename: string
    description: string
    fps: number
    duration_sec: number
  }>({
    queryKey: ['demo-traffic-video'],
    queryFn: async () => (await api.get('/api/traffic/demo-video')).data,
  })
}

// ── Road Intelligence ─────────────────────────────────────────────────────────
export function useRoadSegments() {
  return useQuery<Road[]>({
    queryKey: ['road-intelligence'],
    queryFn: async () => (await api.get('/api/road-intelligence')).data,
  })
}

export function useRoadSegmentDossier(segmentId: number | null) {
  return useQuery<RoadSegmentDossier>({
    queryKey: ['road-intelligence', segmentId],
    queryFn: async () => (await api.get(`/api/road-intelligence/${segmentId}`)).data,
    enabled: segmentId !== null,
  })
}

export function useConflicts(params?: { road_id?: number; risk_level?: string; review_status?: string }) {
  return useQuery<NearMiss[]>({
    queryKey: ['conflicts', params],
    queryFn: async () => (await api.get('/api/conflicts', { params })).data,
  })
}

export function useConflictHotspots() {
  return useQuery<any[]>({
    queryKey: ['conflict-hotspots'],
    queryFn: async () => (await api.get('/api/conflicts/hotspots')).data,
  })
}

export function useReviewConflict() {
  const queryClient = useQueryClient()
  return useMutation<NearMiss, Error, { eventId?: number; conflict_id?: number; review_status?: string; status?: string; notes?: string; reviewer?: string }>({
    mutationFn: async ({ eventId, conflict_id, review_status, status, notes }) => {
      const id = eventId ?? conflict_id
      const st = review_status ?? status ?? 'reviewed'
      return (await api.patch(`/api/conflicts/${id}/review`, { review_status: st, notes })).data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['conflicts'] })
      queryClient.invalidateQueries({ queryKey: ['conflict-hotspots'] })
      queryClient.invalidateQueries({ queryKey: ['dashboard'] })
      queryClient.invalidateQueries({ queryKey: ['road-intelligence'] })
    },
  })
}

// ── Evidence & Audit Logs ─────────────────────────────────────────────────────
export function useEvidence(params?: { road_id?: number; file_type?: string }) {
  return useQuery<EvidenceFile[]>({
    queryKey: ['evidence', params],
    queryFn: async () => (await api.get('/api/evidence', { params })).data,
  })
}

export function useAuditLogs(limit?: number) {
  return useQuery<AuditLog[]>({
    queryKey: ['audit-logs', limit],
    queryFn: async () => (await api.get('/api/audit-logs', { params: { limit } })).data,
  })
}

export function useDecisionAudits() {
  return useQuery<DecisionAuditRecord[]>({
    queryKey: ['decision-audits'],
    queryFn: async () => (await api.get('/api/evidence/decisions')).data,
  })
}

function queryKeyRefetch(client: any, key: string) {
  client.invalidateQueries({ queryKey: [key] })
}

