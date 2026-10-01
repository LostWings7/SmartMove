import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '@/api/client'
import type {
  CitizenComplaint,
  AdminComplaint,
  ComplaintCategory,
  ComplaintEvidence,
  ComplaintStatusHistory,
  CitizenDashboardData,
  AdminDashboardData,
  AppNotification,
  StaffUser,
  User,
  AIHazardCandidate,
  DuplicateCandidate,
} from '@/types'

// ── Citizen Hooks ────────────────────────────────────────────────────────────

export function useCitizenDashboard() {
  return useQuery<CitizenDashboardData>({
    queryKey: ['citizen-dashboard'],
    queryFn: async () => {
      const res = await api.get('/api/citizen/dashboard')
      return res.data
    },
    refetchInterval: 15_000,
  })
}

export function useCategories() {
  return useQuery<ComplaintCategory[]>({
    queryKey: ['complaint-categories'],
    queryFn: async () => {
      const res = await api.get('/api/complaints/categories')
      return res.data
    },
    staleTime: 60_000 * 5,
  })
}

export function useMyComplaints(filters?: { status?: string; category_id?: number; search?: string; sort_order?: string }) {
  return useQuery<CitizenComplaint[]>({
    queryKey: ['my-complaints', filters],
    queryFn: async () => {
      const res = await api.get('/api/complaints/my', { params: filters })
      return res.data
    },
  })
}

export function useComplaintDetail(id: number | string) {
  return useQuery<CitizenComplaint>({
    queryKey: ['complaint-detail', id],
    queryFn: async () => {
      const res = await api.get(`/api/complaints/${id}`)
      return res.data
    },
    enabled: !!id,
  })
}

export function useSubmitComplaint() {
  const qc = useQueryClient()
  return useMutation<
    CitizenComplaint,
    Error,
    { category_id: number; title: string; description: string; location_lat: number; location_lng: number; location_address?: string; priority?: string }
  >({
    mutationFn: async (payload) => {
      const res = await api.post('/api/complaints', payload)
      return res.data
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['my-complaints'] })
      qc.invalidateQueries({ queryKey: ['citizen-dashboard'] })
      qc.invalidateQueries({ queryKey: ['admin-complaints'] })
      qc.invalidateQueries({ queryKey: ['admin-dashboard'] })
    },
  })
}

export function useUploadEvidence() {
  const qc = useQueryClient()
  return useMutation<ComplaintEvidence, Error, { complaintId: number; file: File; notes?: string }>({
    mutationFn: async ({ complaintId, file, notes }) => {
      const formData = new FormData()
      formData.append('file', file)
      if (notes) formData.append('notes', notes)

      const res = await api.post(`/api/complaints/${complaintId}/evidence`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      return res.data
    },
    onSuccess: (_, { complaintId }) => {
      qc.invalidateQueries({ queryKey: ['complaint-detail', complaintId] })
      qc.invalidateQueries({ queryKey: ['admin-complaint-detail', complaintId] })
    },
  })
}

export function useCitizenConfirmResolution() {
  const qc = useQueryClient()
  return useMutation<CitizenComplaint, Error, { complaintId: number; confirmed: boolean; comment?: string }>({
    mutationFn: async ({ complaintId, confirmed, comment }) => {
      const res = await api.post(`/api/complaints/${complaintId}/citizen-confirm`, { confirmed, comment })
      return res.data
    },
    onSuccess: (_, { complaintId }) => {
      qc.invalidateQueries({ queryKey: ['complaint-detail', complaintId] })
      qc.invalidateQueries({ queryKey: ['my-complaints'] })
      qc.invalidateQueries({ queryKey: ['citizen-dashboard'] })
      qc.invalidateQueries({ queryKey: ['admin-complaints'] })
      qc.invalidateQueries({ queryKey: ['admin-dashboard'] })
    },
  })
}

// ── Notifications ─────────────────────────────────────────────────────────────

export function useNotifications() {
  return useQuery<AppNotification[]>({
    queryKey: ['notifications'],
    queryFn: async () => {
      const res = await api.get('/api/notifications')
      return res.data
    },
    refetchInterval: 10_000,
  })
}

export function useUnreadNotificationCount() {
  return useQuery<{ unread_count: number }>({
    queryKey: ['unread-notifications-count'],
    queryFn: async () => {
      const res = await api.get('/api/notifications/unread-count')
      return res.data
    },
    refetchInterval: 10_000,
  })
}

export function useMarkNotificationRead() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async (id: number) => {
      const res = await api.post(`/api/notifications/${id}/read`)
      return res.data
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['notifications'] })
      qc.invalidateQueries({ queryKey: ['unread-notifications-count'] })
    },
  })
}

export function useMarkAllNotificationsRead() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async () => {
      const res = await api.post('/api/notifications/read-all')
      return res.data
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['notifications'] })
      qc.invalidateQueries({ queryKey: ['unread-notifications-count'] })
    },
  })
}

// ── Admin Hooks ──────────────────────────────────────────────────────────────

export function useAdminDashboard() {
  return useQuery<AdminDashboardData>({
    queryKey: ['admin-dashboard'],
    queryFn: async () => {
      const res = await api.get('/api/admin/dashboard')
      return res.data
    },
    refetchInterval: 15_000,
  })
}

export interface AdminComplaintsParams {
  page?: number
  page_size?: number
  status?: string
  category_id?: number
  priority?: string
  assigned_to?: number
  ai_linked?: boolean
  search?: string
  sort_by?: string
  sort_order?: string
}

export function useAdminComplaints(params?: AdminComplaintsParams) {
  return useQuery<{
    items: AdminComplaint[]
    total: number
    page: number
    page_size: number
    total_pages: number
  }>({
    queryKey: ['admin-complaints', params],
    queryFn: async () => {
      const res = await api.get('/api/admin/complaints', { params })
      return res.data
    },
  })
}

export function useAdminComplaintDetail(id: number | string) {
  return useQuery<{
    complaint: AdminComplaint
    ai_hazard_candidates: AIHazardCandidate[]
    duplicate_candidates: DuplicateCandidate[]
  }>({
    queryKey: ['admin-complaint-detail', id],
    queryFn: async () => {
      const res = await api.get(`/api/admin/complaints/${id}`)
      return res.data
    },
    enabled: !!id,
  })
}

export function useUpdateComplaintStatus() {
  const qc = useQueryClient()
  return useMutation<AdminComplaint, Error, { complaintId: number; new_status: string; comment?: string }>({
    mutationFn: async ({ complaintId, new_status, comment }) => {
      const res = await api.post(`/api/admin/complaints/${complaintId}/status`, { new_status, comment })
      return res.data
    },
    onSuccess: (_, { complaintId }) => {
      qc.invalidateQueries({ queryKey: ['admin-complaint-detail', complaintId] })
      qc.invalidateQueries({ queryKey: ['admin-complaints'] })
      qc.invalidateQueries({ queryKey: ['admin-dashboard'] })
    },
  })
}

export function useAssignComplaint() {
  const qc = useQueryClient()
  return useMutation<
    AdminComplaint,
    Error,
    { complaintId: number; assigned_to?: number; assigned_department?: string; notes?: string }
  >({
    mutationFn: async ({ complaintId, ...payload }) => {
      const res = await api.post(`/api/admin/complaints/${complaintId}/assign`, payload)
      return res.data
    },
    onSuccess: (_, { complaintId }) => {
      qc.invalidateQueries({ queryKey: ['admin-complaint-detail', complaintId] })
      qc.invalidateQueries({ queryKey: ['admin-complaints'] })
    },
  })
}

export function useAddPublicResponse() {
  const qc = useQueryClient()
  return useMutation<AdminComplaint, Error, { complaintId: number; public_response: string }>({
    mutationFn: async ({ complaintId, public_response }) => {
      const res = await api.post(`/api/admin/complaints/${complaintId}/public-response`, { public_response })
      return res.data
    },
    onSuccess: (_, { complaintId }) => {
      qc.invalidateQueries({ queryKey: ['admin-complaint-detail', complaintId] })
      qc.invalidateQueries({ queryKey: ['admin-complaints'] })
    },
  })
}

export function useAddInternalNote() {
  const qc = useQueryClient()
  return useMutation<AdminComplaint, Error, { complaintId: number; internal_note: string }>({
    mutationFn: async ({ complaintId, internal_note }) => {
      const res = await api.post(`/api/admin/complaints/${complaintId}/internal-note`, { internal_note })
      return res.data
    },
    onSuccess: (_, { complaintId }) => {
      qc.invalidateQueries({ queryKey: ['admin-complaint-detail', complaintId] })
    },
  })
}

export function useLinkHazard() {
  const qc = useQueryClient()
  return useMutation<AdminComplaint, Error, { complaintId: number; hazard_id: number; notes?: string }>({
    mutationFn: async ({ complaintId, hazard_id, notes }) => {
      const res = await api.post(`/api/admin/complaints/${complaintId}/link-hazard`, { hazard_id, notes })
      return res.data
    },
    onSuccess: (_, { complaintId }) => {
      qc.invalidateQueries({ queryKey: ['admin-complaint-detail', complaintId] })
      qc.invalidateQueries({ queryKey: ['admin-complaints'] })
      qc.invalidateQueries({ queryKey: ['roads'] })
      qc.invalidateQueries({ queryKey: ['hazards'] })
    },
  })
}

export function useCreateHazard() {
  const qc = useQueryClient()
  return useMutation<AdminComplaint, Error, { complaintId: number; road_id?: number; severity?: string; notes?: string }>({
    mutationFn: async ({ complaintId, ...payload }) => {
      const res = await api.post(`/api/admin/complaints/${complaintId}/create-hazard`, payload)
      return res.data
    },
    onSuccess: (_, { complaintId }) => {
      qc.invalidateQueries({ queryKey: ['admin-complaint-detail', complaintId] })
      qc.invalidateQueries({ queryKey: ['admin-complaints'] })
      qc.invalidateQueries({ queryKey: ['roads'] })
      qc.invalidateQueries({ queryKey: ['hazards'] })
      qc.invalidateQueries({ queryKey: ['repair-priority'] })
    },
  })
}

export function useLinkRepair() {
  const qc = useQueryClient()
  return useMutation<AdminComplaint, Error, { complaintId: number; repair_id: number; notes?: string }>({
    mutationFn: async ({ complaintId, repair_id, notes }) => {
      const res = await api.post(`/api/admin/complaints/${complaintId}/link-repair`, { repair_id, notes })
      return res.data
    },
    onSuccess: (_, { complaintId }) => {
      qc.invalidateQueries({ queryKey: ['admin-complaint-detail', complaintId] })
    },
  })
}

export function useStaffUsers() {
  return useQuery<StaffUser[]>({
    queryKey: ['staff-users'],
    queryFn: async () => {
      const res = await api.get('/api/admin/staff')
      return res.data
    },
  })
}

export function useAdminUsers(params?: { page?: number; page_size?: number; role?: string; is_active?: boolean; search?: string }) {
  return useQuery<{
    items: User[]
    total: number
    page: number
    page_size: number
    total_pages: number
  }>({
    queryKey: ['admin-users', params],
    queryFn: async () => {
      const res = await api.get('/api/admin/users', { params })
      return res.data
    },
  })
}

export function useUpdateUserRole() {
  const qc = useQueryClient()
  return useMutation<User, Error, { userId: number; role: string }>({
    mutationFn: async ({ userId, role }) => {
      const res = await api.patch(`/api/admin/users/${userId}/role`, { role })
      return res.data
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['admin-users'] })
      qc.invalidateQueries({ queryKey: ['staff-users'] })
    },
  })
}

export function useUpdateUserStatus() {
  const qc = useQueryClient()
  return useMutation<User, Error, { userId: number; is_active: boolean }>({
    mutationFn: async ({ userId, is_active }) => {
      const res = await api.patch(`/api/admin/users/${userId}/status`, { is_active })
      return res.data
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['admin-users'] })
      qc.invalidateQueries({ queryKey: ['staff-users'] })
    },
  })
}
