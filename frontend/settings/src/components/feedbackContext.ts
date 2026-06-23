import { createContext, useContext } from 'react'

export type ToastType = 'info' | 'success' | 'error'

export interface ConfirmOptions {
  title?: string
  message: string
  confirmLabel?: string
  cancelLabel?: string
  danger?: boolean
}

export interface PromptOptions {
  title?: string
  message?: string
  label?: string
  placeholder?: string
  initialValue?: string
  confirmLabel?: string
}

export interface FeedbackApi {
  /** transient corner notification - replaces alert() for non-blocking info/errors */
  toast: (message: string, type?: ToastType) => void
  /** modal yes/no - replaces window.confirm(); resolves false on cancel/Esc */
  confirm: (opts: ConfirmOptions) => Promise<boolean>
  /** modal text entry - replaces window.prompt(); resolves null on cancel/Esc */
  prompt: (opts: PromptOptions) => Promise<string | null>
}

export const FeedbackContext = createContext<FeedbackApi>({
  toast: () => {},
  confirm: async () => false,
  prompt: async () => null,
})

export function useFeedback(): FeedbackApi {
  return useContext(FeedbackContext)
}
