import type { GpuStat } from '../api/client'

function barColor(pct: number): string {
  if (pct >= 90) return '#dc2626'
  if (pct >= 70) return '#f59e0b'
  return '#16a34a'
}

function Bar({ label, pct, sublabel }: { label: string; pct: number; sublabel: string }) {
  return (
    <div className="gpu-bar">
      <div className="gpu-bar-head">
        <span>{label}</span>
        <span className="gpu-bar-sublabel">{sublabel}</span>
      </div>
      <div className="gpu-bar-track">
        <div
          className="gpu-bar-fill"
          style={{ width: `${Math.min(pct, 100)}%`, background: barColor(pct) }}
        />
      </div>
    </div>
  )
}

export function GpuGauge({ gpu }: { gpu: GpuStat }) {
  const memPct = gpu.memory_total_mb > 0 ? (100 * gpu.memory_used_mb) / gpu.memory_total_mb : 0
  return (
    <div className="gpu-gauge">
      <div className="gpu-gauge-title">
        GPU {gpu.gpu_id} — {gpu.name}
        {gpu.temperature_c != null && <span className="gpu-temp"> · {gpu.temperature_c}°C</span>}
        {gpu.power_w != null && <span className="gpu-temp"> · {gpu.power_w}W</span>}
      </div>
      <Bar label="Utilization" pct={gpu.utilization_pct} sublabel={`${gpu.utilization_pct}%`} />
      <Bar
        label="VRAM"
        pct={memPct}
        sublabel={`${gpu.memory_used_mb.toLocaleString()} / ${gpu.memory_total_mb.toLocaleString()} MB`}
      />
    </div>
  )
}
