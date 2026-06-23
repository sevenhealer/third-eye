import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { AdvancedModeProvider } from './components/AdvancedToggle'
import { AppShell } from './AppShell'
import { RequireAuth } from './RequireAuth'
import { LoginPage } from './pages/LoginPage'
import { DashboardPage } from './pages/DashboardPage'
import { ZonesPage } from './pages/ZonesPage'
import { CandidatesPage } from './pages/CandidatesPage'
import { AlertsPage } from './pages/AlertsPage'
import { CamerasPage } from './pages/CamerasPage'
import { HardwarePage } from './pages/HardwarePage'
import { SystemPage } from './pages/SystemPage'

function App() {
  return (
    <BrowserRouter basename="/settings">
      <AdvancedModeProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route element={<RequireAuth />}>
            <Route element={<AppShell />}>
              <Route path="/" element={<DashboardPage />} />
              <Route path="/zones" element={<ZonesPage />} />
              <Route path="/candidates" element={<CandidatesPage />} />
              <Route path="/alerts" element={<AlertsPage />} />
              <Route path="/cameras" element={<CamerasPage />} />
              <Route path="/hardware" element={<HardwarePage />} />
              <Route path="/system" element={<SystemPage />} />
            </Route>
          </Route>
        </Routes>
      </AdvancedModeProvider>
    </BrowserRouter>
  )
}

export default App
