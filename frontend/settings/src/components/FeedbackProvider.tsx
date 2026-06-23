import { useCallback, useMemo, useState, type ReactNode } from 'react'
import {
  FeedbackContext,
  type ConfirmOptions,
  type PromptOptions,
  type ToastType,
} from './feedbackContext'
import { Modal } from './Modal'

interface Toast {
  id: number
  message: string
  type: ToastType
}
type ConfirmState = ConfirmOptions & { resolve: (v: boolean) => void }
type PromptState = PromptOptions & { resolve: (v: string | null) => void }

const TOAST_TTL_MS = 4500
let nextToastId = 1

/**
 * App-wide feedback: a toast stack plus promise-based confirm/prompt
 * dialogs, so pages can `await confirm(...)` / `await prompt(...)` and
 * `toast(...)` instead of the blocking native window.confirm/prompt/alert
 * that don't match the dark admin-panel styling.
 */
export function FeedbackProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const [confirmState, setConfirmState] = useState<ConfirmState | null>(null)
  const [promptState, setPromptState] = useState<PromptState | null>(null)
  const [promptValue, setPromptValue] = useState('')

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  const toast = useCallback(
    (message: string, type: ToastType = 'info') => {
      const id = nextToastId++
      setToasts((prev) => [...prev, { id, message, type }])
      setTimeout(() => dismiss(id), TOAST_TTL_MS)
    },
    [dismiss],
  )

  const confirm = useCallback((opts: ConfirmOptions) => {
    return new Promise<boolean>((resolve) => setConfirmState({ ...opts, resolve }))
  }, [])

  const prompt = useCallback((opts: PromptOptions) => {
    setPromptValue(opts.initialValue ?? '')
    return new Promise<string | null>((resolve) => setPromptState({ ...opts, resolve }))
  }, [])

  const api = useMemo(() => ({ toast, confirm, prompt }), [toast, confirm, prompt])

  function resolveConfirm(value: boolean) {
    confirmState?.resolve(value)
    setConfirmState(null)
  }
  function resolvePrompt(value: string | null) {
    promptState?.resolve(value)
    setPromptState(null)
  }

  return (
    <FeedbackContext.Provider value={api}>
      {children}

      <div className="toast-stack">
        {toasts.map((t) => (
          <div key={t.id} className={`toast toast-${t.type}`} role="status">
            <span>{t.message}</span>
            <button className="toast-close" onClick={() => dismiss(t.id)} aria-label="Dismiss">
              ×
            </button>
          </div>
        ))}
      </div>

      {confirmState && (
        <Modal
          title={confirmState.title ?? 'Please confirm'}
          width={360}
          onClose={() => resolveConfirm(false)}
        >
          <p className="dialog-message">{confirmState.message}</p>
          <div className="dialog-actions">
            <button onClick={() => resolveConfirm(false)}>{confirmState.cancelLabel ?? 'Cancel'}</button>
            <button
              className={confirmState.danger ? 'btn-danger' : 'btn-primary'}
              onClick={() => resolveConfirm(true)}
            >
              {confirmState.confirmLabel ?? 'Confirm'}
            </button>
          </div>
        </Modal>
      )}

      {promptState && (
        <Modal
          title={promptState.title ?? 'Enter a value'}
          width={360}
          onClose={() => resolvePrompt(null)}
        >
          <form
            onSubmit={(e) => {
              e.preventDefault()
              resolvePrompt(promptValue.trim() || null)
            }}
          >
            {promptState.message && <p className="dialog-message">{promptState.message}</p>}
            <label className="dialog-label">
              {promptState.label}
              <input
                autoFocus
                value={promptValue}
                placeholder={promptState.placeholder}
                onChange={(e) => setPromptValue(e.target.value)}
              />
            </label>
            <div className="dialog-actions">
              <button type="button" onClick={() => resolvePrompt(null)}>
                Cancel
              </button>
              <button type="submit" className="btn-primary" disabled={!promptValue.trim()}>
                {promptState.confirmLabel ?? 'OK'}
              </button>
            </div>
          </form>
        </Modal>
      )}
    </FeedbackContext.Provider>
  )
}
