import { useEffect, useRef } from 'react'

/**
 * Run `callback` once on mount and then every `intervalMs` - but pause
 * while the tab is hidden, and fire an immediate refresh the moment it
 * becomes visible again. Replaces the repeated
 * `useEffect(() => { fn(); setInterval(fn, n) })` pattern that kept hitting
 * the API every few seconds even on a backgrounded tab.
 */
export function usePolling(callback: () => void, intervalMs: number): void {
  const callbackRef = useRef(callback)
  useEffect(() => {
    callbackRef.current = callback
  }, [callback])

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null

    const tick = () => callbackRef.current()

    const start = () => {
      if (timer !== null) return
      tick()   // immediate refresh so a resumed/just-mounted tab isn't stale
      timer = setInterval(tick, intervalMs)
    }
    const stop = () => {
      if (timer !== null) {
        clearInterval(timer)
        timer = null
      }
    }

    const onVisibility = () => {
      if (document.hidden) stop()
      else start()
    }

    if (!document.hidden) start()
    document.addEventListener('visibilitychange', onVisibility)

    return () => {
      stop()
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [intervalMs])
}
