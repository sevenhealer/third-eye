import { useAdvancedMode } from '../components/advancedModeContext'

export function SystemPage() {
  const { advanced } = useAdvancedMode()

  return (
    <div>
      <div className="page-head">
        <h2>System</h2>
      </div>

      <div className="panel">
        <h3>Links</h3>
        <ul className="link-list">
          <li>
            <a href="http://localhost:3000" target="_blank" rel="noreferrer">
              Grafana
            </a>{' '}
            — historical metrics (E10-S02: GPU VRAM, inference latency, Kafka lag)
          </li>
          {advanced && (
            <>
              <li>
                <a href="/docs" target="_blank" rel="noreferrer">
                  API docs (Swagger)
                </a>
              </li>
              <li>
                <a href="/metrics" target="_blank" rel="noreferrer">
                  Prometheus /metrics
                </a>
              </li>
              <li>
                <a href="/dashboard-legacy/" target="_blank" rel="noreferrer">
                  Legacy dashboard
                </a>{' '}
                — the original vanilla-JS version, kept as a rollback path during burn-in
              </li>
            </>
          )}
        </ul>
      </div>

      {!advanced && (
        <p className="hint" style={{ marginTop: 18 }}>
          Turn on Advanced mode (top right) for API docs, raw metrics, and per-camera crash history.
        </p>
      )}
    </div>
  )
}
