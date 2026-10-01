import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { AuthProvider } from '@/context/AuthContext'
import { VoiceProvider } from '@/context/VoiceContext'
import FridayVoiceWidget from '@/components/voice/FridayVoiceWidget'
import ProtectedRoute from '@/components/auth/ProtectedRoute'

// Core SafeCity Pages
import Overview from '@/pages/Overview'
import SafeCityMap from '@/pages/SafeCityMap'
import AIVision from '@/pages/AIVision'
import DangerZones from '@/pages/DangerZones'
import RepairIntelligence from '@/pages/RepairIntelligence'
import SaferRoutes from '@/pages/SaferRoutes'
import SmartJunction from '@/pages/SmartJunction'
import DigitalTwin from '@/pages/DigitalTwin'
import Analytics from '@/pages/Analytics'
import CitizenReports from '@/pages/CitizenReports'
import InterventionImpact from '@/pages/InterventionImpact'
import Settings from '@/pages/Settings'
import LandingPage from '@/pages/LandingPage'
import SmartJunctionBillboard from '@/pages/SmartJunctionBillboard'
import RoadIntelligence from '@/pages/RoadIntelligence'
import TrafficConflicts from '@/pages/TrafficConflicts'
import EvidenceAudit from '@/pages/EvidenceAudit'
import Privacy from '@/pages/Privacy'

// Citizen & Admin Portal Pages
import Login from '@/pages/Login'
import Register from '@/pages/Register'
import CitizenDashboard from '@/pages/CitizenDashboard'
import CreateComplaint from '@/pages/CreateComplaint'
import MyComplaints from '@/pages/MyComplaints'
import ComplaintDetail from '@/pages/ComplaintDetail'
import NotificationsPage from '@/pages/NotificationsPage'
import CitizenProfile from '@/pages/CitizenProfile'
import AdminDashboard from '@/pages/AdminDashboard'
import AdminComplaints from '@/pages/AdminComplaints'
import AdminComplaintDetail from '@/pages/AdminComplaintDetail'
import UserManagement from '@/pages/UserManagement'

export default function App() {
  return (
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <AuthProvider>
        <VoiceProvider>
          <Routes>
          {/* Public Auth Routes */}
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/landing" element={<LandingPage />} />
          <Route path="/privacy" element={<Privacy />} />

          {/* Citizen Portal (Protected) */}
          <Route
            path="/citizen/dashboard"
            element={
              <ProtectedRoute allowedRoles={['CITIZEN', 'ADMIN', 'SUPERUSER']}>
                <CitizenDashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/citizen/complaints"
            element={
              <ProtectedRoute allowedRoles={['CITIZEN', 'ADMIN', 'SUPERUSER']}>
                <MyComplaints />
              </ProtectedRoute>
            }
          />
          <Route
            path="/citizen/complaints/new"
            element={
              <ProtectedRoute allowedRoles={['CITIZEN', 'ADMIN', 'SUPERUSER']}>
                <CreateComplaint />
              </ProtectedRoute>
            }
          />
          <Route
            path="/citizen/complaints/:id"
            element={
              <ProtectedRoute allowedRoles={['CITIZEN', 'ADMIN', 'SUPERUSER']}>
                <ComplaintDetail />
              </ProtectedRoute>
            }
          />
          <Route
            path="/citizen/notifications"
            element={
              <ProtectedRoute allowedRoles={['CITIZEN', 'INSPECTOR', 'ENGINEER', 'ADMIN', 'SUPERUSER']}>
                <NotificationsPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/citizen/profile"
            element={
              <ProtectedRoute allowedRoles={['CITIZEN', 'INSPECTOR', 'ENGINEER', 'ADMIN', 'SUPERUSER']}>
                <CitizenProfile />
              </ProtectedRoute>
            }
          />

          {/* Admin & Municipal Staff Portal (Protected) */}
          <Route
            path="/admin/dashboard"
            element={
              <ProtectedRoute allowedRoles={['INSPECTOR', 'ENGINEER', 'ADMIN', 'SUPERUSER']}>
                <AdminDashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/admin/complaints"
            element={
              <ProtectedRoute allowedRoles={['INSPECTOR', 'ENGINEER', 'ADMIN', 'SUPERUSER']}>
                <AdminComplaints />
              </ProtectedRoute>
            }
          />
          <Route
            path="/admin/complaints/:id"
            element={
              <ProtectedRoute allowedRoles={['INSPECTOR', 'ENGINEER', 'ADMIN', 'SUPERUSER']}>
                <AdminComplaintDetail />
              </ProtectedRoute>
            }
          />
          <Route
            path="/admin/users"
            element={
              <ProtectedRoute allowedRoles={['ADMIN', 'SUPERUSER']}>
                <UserManagement />
              </ProtectedRoute>
            }
          />

          {/* SafeCity Engineering & Intelligence Platforms */}
          <Route path="/" element={<Overview />} />
          <Route path="/road-intelligence" element={<RoadIntelligence />} />
          <Route path="/traffic-conflicts" element={<TrafficConflicts />} />
          <Route path="/evidence" element={<EvidenceAudit />} />
          <Route path="/map" element={<SafeCityMap />} />
          <Route path="/ai-vision" element={<AIVision />} />
          <Route path="/danger-zones" element={<DangerZones />} />
          <Route path="/repair" element={<RepairIntelligence />} />
          <Route path="/routes" element={<SaferRoutes />} />
          <Route path="/junction" element={<SmartJunction />} />
          <Route path="/junction-display" element={<SmartJunctionBillboard />} />
          <Route path="/digital-twin" element={<DigitalTwin />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/reports" element={<CitizenReports />} />
          <Route path="/interventions" element={<InterventionImpact />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
        <FridayVoiceWidget />
      </VoiceProvider>
    </AuthProvider>
  </BrowserRouter>
  )
}
