import { NavLink, Outlet } from 'react-router-dom'
import { AdvancedToggle } from './components/AdvancedToggle'
import { Icon, type IconName } from './components/Icon'

const NAV_ITEMS: { to: string; label: string; icon: IconName }[] = [
  { to: '/', label: 'Dashboard', icon: 'dashboard' },
  { to: '/zones', label: 'Zones', icon: 'zones' },
  { to: '/candidates', label: 'Candidates', icon: 'candidates' },
  { to: '/enroll', label: 'Enroll', icon: 'enroll' },
  { to: '/alerts', label: 'Alerts', icon: 'alerts' },
  { to: '/cameras', label: 'Cameras', icon: 'cameras' },
  { to: '/hardware', label: 'Hardware', icon: 'hardware' },
  { to: '/system', label: 'System', icon: 'system' },
]

export function AppShell() {
  return (
    <>
      <header className="settings-header">
        <div className="brand">
          <span className="brand-mark">
            <Icon name="eye" />
          </span>
          <div>
            <h1>Third-Eye</h1>
            <div className="brand-sub">Visual Intelligence</div>
          </div>
        </div>
        <nav className="tabs">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === '/'}
              className={({ isActive }) => (isActive ? 'active' : '')}
            >
              <Icon name={item.icon} />
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
