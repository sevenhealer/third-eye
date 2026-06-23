import { NavLink, Outlet } from 'react-router-dom'
import { AdvancedToggle } from './components/AdvancedToggle'

const NAV_ITEMS: { to: string; label: string }[] = [
  { to: '/', label: 'Dashboard' },
  { to: '/zones', label: 'Zones' },
  { to: '/candidates', label: 'Candidates' },
  { to: '/alerts', label: 'Alerts' },
  { to: '/cameras', label: 'Cameras' },
  { to: '/hardware', label: 'Hardware' },
  { to: '/system', label: 'System' },
]

export function AppShell() {
  return (
    <>
      <header className="settings-header">
        <h1>Third-Eye</h1>
        <nav className="tabs">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) => (isActive ? 'active' : '')}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <AdvancedToggle />
      </header>
      <main className="settings-main">
        <Outlet />
      </main>
    </>
  )
}
