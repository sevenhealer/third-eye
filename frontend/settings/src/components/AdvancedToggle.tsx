import { useEffect, useState, type ReactNode } from 'react'
import { AdvancedModeContext, useAdvancedMode } from './advancedModeContext'

const STORAGE_KEY = 'te_settings_advanced'

// A UI preference, not auth state - persisted to localStorage (survives
// logout/login) rather than sessionStorage (which the token uses).
export function AdvancedModeProvider({ children }: { children: ReactNode }) {
  const [advanced, setAdvanced] = useState(() => localStorage.getItem(STORAGE_KEY) === '1')

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, advanced ? '1' : '0')
  }, [advanced])

  return (
    <AdvancedModeContext.Provider value={{ advanced, setAdvanced }}>
      {children}
    </AdvancedModeContext.Provider>
  )
}

export function AdvancedToggle() {
  const { advanced, setAdvanced } = useAdvancedMode()
  return (
    <label className="advanced-toggle">
      <input type="checkbox" checked={advanced} onChange={(e) => setAdvanced(e.target.checked)} />
      Advanced mode
    </label>
  )
}
