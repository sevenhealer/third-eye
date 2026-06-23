import { useEffect, useState } from 'react'
import { wsUrl } from '../api/client'

interface AlertMessage {
  alert_id?: string
  event_type?: string
  alert_type?: string
  severity?: string
  zone_id?: string
  camera_id?: string
  description?: string
}

const MAX_ALERTS = 100

export function AlertsPage() {
  const [alerts, setAlerts] = useState<AlertMessage[]>([])
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    const ws = new WebSocket(wsUrl('/api/v1/alerts/ws'))
    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onmessage = (ev) => {
      const a = JSON.parse(ev.data) as AlertMessage
      setAlerts((prev) => [a, ...prev].slice(0, MAX_ALERTS))
    }
    return () => ws.close()
  }, [])

  return (
    <div>
      <div className="page-head">
        <h2>Live Alerts</h2>
        <span className={`status-dot ${connected ? 'dot-online' : 'dot-offline'}`} />
      </div>

      <ul className="alerts-list panel">
        {!alerts.length && (
          <li className="empty">{connected ? 'Connected — waiting for alerts…' : 'Disconnected.'}</li>
        )}
        {alerts.map((a, i) => {
          const severity = a.severity || 'LOW'
          return (
            <li key={a.alert_id || i}>
              <span>
                {a.event_type || a.alert_type || 'EVENT'} — {a.zone_id || ''}
                {a.description ? ` (${a.description})` : ''}
              </span>
              <span className={`sev-${severity}`}>{severity}</span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
