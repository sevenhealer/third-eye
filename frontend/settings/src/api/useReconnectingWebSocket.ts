import { useEffect, useRef, useState } from 'react'
import { refreshAccessToken, wsUrl } from './client'

// Server closes the socket with this code when the ?token= access token is
// missing/expired/not-an-access-token (see alerts.py / hardware.py). A
// reconnect must get a fresh token first, or it just gets kicked again.
const AUTH_CLOSE_CODE = 4401
const MAX_BACKOFF_MS = 15_000

/**
 * A WebSocket that survives the things that actually happen to this app:
 * the API server restarting (constant in dev), a network blip, and the
 * 15-minute access token expiring under a long-lived connection. The raw
 * `new WebSocket()` in a useEffect that this replaces died silently on any
 * of those until a full page reload.
 *
 * `onMessage` receives the already-JSON-parsed payload; a frame that fails
 * to parse is dropped rather than throwing inside the socket handler.
 * Returns the live connection status for a header dot.
 */
export function useReconnectingWebSocket(
  path: string,
  onMessage: (data: unknown) => void,
): boolean {
  const [connected, setConnected] = useState(false)
  // Keep the latest callback without making it a dependency - otherwise an
  // inline `onMessage` (new identity every render) would tear down and
  // rebuild the socket on every parent re-render (e.g. the alerts list
  // growing), which is exactly the reconnect storm we're trying to avoid.
  const onMessageRef = useRef(onMessage)
  useEffect(() => {
    onMessageRef.current = onMessage
  }, [onMessage])

  useEffect(() => {
    let ws: WebSocket | null = null
    let attempt = 0
    let timer: ReturnType<typeof setTimeout> | null = null
    let disposed = false   // component unmounted - stop reconnecting

    const connect = () => {
      if (disposed) return
      ws = new WebSocket(wsUrl(path))

      ws.onopen = () => {
        attempt = 0
        setConnected(true)
      }

      ws.onmessage = (ev) => {
        let parsed: unknown
        try {
          parsed = JSON.parse(ev.data)
        } catch {
          return   // malformed frame - skip, don't crash the handler
        }
        onMessageRef.current(parsed)
      }

      ws.onerror = () => ws?.close()

      ws.onclose = async (ev) => {
        setConnected(false)
        if (disposed) return
        // Expired access token: refresh once so the reconnect's wsUrl()
        // picks up a fresh one. If the refresh token is also dead, the
        // page's REST polling hits its own 401 and bounces to /login.
        if (ev.code === AUTH_CLOSE_CODE) {
          await refreshAccessToken()
          if (disposed) return
        }
        const delay = Math.min(1000 * 2 ** attempt, MAX_BACKOFF_MS)
        attempt += 1
        timer = setTimeout(connect, delay)
      }
    }

    connect()

    return () => {
      disposed = true
      if (timer) clearTimeout(timer)
      // Drop our handlers before closing so the teardown close doesn't
      // schedule a reconnect.
      if (ws) {
        ws.onclose = null
        ws.close()
      }
    }
  }, [path])

  return connected
}
