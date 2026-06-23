import { useEffect, useRef, type ReactNode } from 'react'

interface Props {
  title: ReactNode
  onClose: () => void
  children: ReactNode
  width?: number
  /** extra class on the inner box (e.g. 'camera-form', 'log-modal') */
  className?: string
  /** header actions rendered left of the Close button (e.g. a Refresh) */
  headerExtra?: ReactNode
}

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'

/**
 * One modal shell for the whole app: closes on Escape and on overlay
 * click, traps Tab focus inside the dialog, moves focus to the first
 * field on open, and restores it to the previously-focused element on
 * close. Replaces five hand-rolled overlay divs that did none of that.
 */
export function Modal({ title, onClose, children, width, className, headerExtra }: Props) {
  const boxRef = useRef<HTMLDivElement>(null)
  const onCloseRef = useRef(onClose)
  useEffect(() => {
    onCloseRef.current = onClose
  }, [onClose])

  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null

    // Focus the first meaningful field (skip the Close button so typing
    // starts in the form, not on dismiss).
    const focusables = boxRef.current?.querySelectorAll<HTMLElement>(FOCUSABLE)
    const firstField = Array.from(focusables ?? []).find(
      (el) => !el.classList.contains('modal-close'),
    )
    firstField?.focus()

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation()
        onCloseRef.current()
        return
      }
      if (e.key !== 'Tab') return
      const items = boxRef.current?.querySelectorAll<HTMLElement>(FOCUSABLE)
      if (!items || items.length === 0) return
      const list = Array.from(items)
      const first = list[0]
      const last = list[list.length - 1]
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault()
        last.focus()
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault()
        first.focus()
      }
    }

    document.addEventListener('keydown', onKeyDown, true)
    return () => {
      document.removeEventListener('keydown', onKeyDown, true)
      previouslyFocused?.focus?.()
    }
  }, [])

  return (
    <div
      className="modal-overlay"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        ref={boxRef}
        className={`modal-box${className ? ` ${className}` : ''}`}
        style={width ? { width } : undefined}
        role="dialog"
        aria-modal="true"
      >
        <div className="modal-head">
          <strong>{title}</strong>
          <div className="modal-head-actions">
            {headerExtra}
            <button className="modal-close" onClick={onClose} aria-label="Close">
              Close
            </button>
          </div>
        </div>
        {children}
      </div>
    </div>
  )
}
