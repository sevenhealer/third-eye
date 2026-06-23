import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { AdvancedModeProvider } from './components/AdvancedToggle'
import { FeedbackProvider } from './components/FeedbackProvider'
import { AppShell } from './AppShell'
import { RequireAuth } from './RequireAuth'
import { LoginPage } from './pages/LoginPage'
import { DashboardPage } from './pages/DashboardPage'
import { ZonesPage } from './pages/ZonesPage'
import { CandidatesPage } from './pages/CandidatesPage'
import { EnrollPage } from './pages/EnrollPage'
import { AlertsPage } from './pages/AlertsPage'
import { CamerasPage } from './pages/CamerasPage'
import { HardwarePage } from './pages/HardwarePage'
import { SystemPage } from './pages/SystemPage'

function App() {
  return (
    <BrowserRouter basename="/settings">
      <AdvancedModeProvider>
        <FeedbackProvider>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route element={<RequireAuth />}>
              <Route element={<AppShell />}>
                <Route path="/" element={<DashboardPage />} />
                <Route path="/zones" element={<ZonesPage />} />
                <Route path="/candidates" element={<CandidatesPage />} />
                <Route path="/enroll" element={<EnrollPage />} />
                <Route path="/alerts" element={<AlertsPage />} />
                <Route path="/cameras" element={<CamerasPage />} />
                <Route path="/hardware" element={<HardwarePage />} />
                <Route path="/system" element={<SystemPage />} />
              </Route>
            </Route>
          </Routes>
        </FeedbackProvider>
      </AdvancedModeProvider>
    </BrowserRouter>
  )
}

export default App
