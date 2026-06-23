import { createContext, useContext } from 'react'

export const AdvancedModeContext = createContext<{ advanced: boolean; setAdvanced: (v: boolean) => void }>({
  advanced: false,
  setAdvanced: () => {},
})

export function useAdvancedMode() {
  return useContext(AdvancedModeContext)
}
