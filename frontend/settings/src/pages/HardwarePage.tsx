import { useEffect, useState } from 'react'
import { type GpuStat, type SystemStats, getSystemStats, wsUrl } from '../api/client'
import { GpuGauge } from '../components/GpuGauge'

export function HardwarePage() {
  const [gpus, setGpus] = useState<GpuStat[]>([])
  const [system, setSystem] = useState<SystemStats | null>(null)
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    const ws = new WebSocket(wsUrl('/api/v1/hardware/gpu-stats-ws'))
    ws.onopen = () => setConnected(true)
    ws.onclose = () => setConnected(false)
    ws.onmessage = (ev) => {
      const data = JSON.parse(ev.data) as { gpus: GpuStat[] }
      setGpus(data.gpus)
    }
    return () => ws.close()
  }, [])

  useEffect(() => {
    const fetchSystem = () => getSystemStats().then(setSystem).catch(() => {})
    fetchSystem()
    const interval = setInterval(fetchSystem, 5000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div>
      <div className="page-head">
        <h2>Hardware</h2>
        <span className={`status-dot ${connected ? 'dot-online' : 'dot-offline'}`} />
      </div>

      {!gpus.length && <div className="empty">No GPU detected — running CPU-only.</div>}
      <div className="gpu-grid">
        {gpus.map((g) => (
          <GpuGauge key={g.gpu_id} gpu={g} />
        ))}
      </div>

      {system && (
        <div className="panel" style={{ marginTop: 18 }}>
          <h3>System</h3>
          <div className="form-grid-2">
            <div>CPU: {system.cpu_pct}%</div>
            <div>
              RAM: {system.memory_used_gb} / {system.memory_total_gb} GB
            </div>
            {system.disks.map((d) => (
              <div key={d.path}>
                {d.path}: {d.used_gb} / {d.total_gb} GB
              </div>
            ))}
          </div>
        </div>
      )}

      <p className="hint" style={{ marginTop: 18 }}>
        For historical trends (VRAM over time, inference latency, Kafka consumer lag), see the{' '}
        <a href="http://localhost:3000" target="_blank" rel="noreferrer">
          Grafana dashboard
        </a>{' '}
        — this page is a live at-a-glance view, not a replacement for it.
      </p>
    </div>
  )
}
