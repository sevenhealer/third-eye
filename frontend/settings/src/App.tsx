import { useEffect, useState } from 'react'
import { getToken } from './api/client'
import { AdvancedModeProvider, AdvancedToggle } from './components/AdvancedToggle'
import { CamerasPage } from './pages/CamerasPage'
import { HardwarePage } from './pages/HardwarePage'
import { SystemPage } from './pages/SystemPage'

type Tab = 'cameras' | 'hardware' | 'system'

function App() {
  const [tab, setTab] = useState<Tab>('cameras')
  const hasToken = getToken() !== null

  useEffect(() => {
    if (!hasToken) {
      window.location.href = '/dashboard/'
    }
  }, [hasToken])

  if (!hasToken) {
    return <div className="empty">Redirecting to login…</div>
  }

  return (
    <AdvancedModeProvider>
      <header className="settings-header">
        <h1>Third-Eye Settings</h1>
        <nav className="tabs">
          <button className={tab === 'cameras' ? 'active' : ''} onClick={() => setTab('cameras')}>
            Cameras
          </button>
          <button className={tab === 'hardware' ? 'active' : ''} onClick={() => setTab('hardware')}>
            Hardware
          </button>
          <button className={tab === 'system' ? 'active' : ''} onClick={() => setTab('system')}>
            System
          </button>
        </nav>
        <AdvancedToggle />
      </header>
      <main className="settings-main">
        {tab === 'cameras' && <CamerasPage />}
        {tab === 'hardware' && <HardwarePage />}
        {tab === 'system' && <SystemPage />}
      </main>
    </AdvancedModeProvider>
  )
}

export default App
