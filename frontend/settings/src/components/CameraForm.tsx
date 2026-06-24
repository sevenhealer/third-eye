import { useState } from 'react'
import type { CameraDetail, CameraLaunchArgs, GpuStat, NewCameraInput } from '../api/client'
import { useAdvancedMode } from './advancedModeContext'
import { Modal } from './Modal'

const DEFAULT_LAUNCH_ARGS: CameraLaunchArgs = {
  fps: 15,
  det_size: 0,
  det_thresh: 0.6,
  reid_thresh: 0.45,
  min_face: 24,
  recognize: false,
  persist_events: false,
  cpu: false,
  enforce_liveness: false,
  collect_antispoofing: false,
}

// For create: the full NewCameraInput. For edit: only the fields that
// actually changed (stream_url omitted entirely if the operator left it
// blank, rather than overwriting a real RTSP URL with an empty string).
export type CameraFormSubmit = NewCameraInput | Partial<NewCameraInput>

interface Props {
  existing?: CameraDetail
  gpus: GpuStat[]
  onSubmit: (input: CameraFormSubmit) => Promise<void>
  onCancel: () => void
}

export function CameraForm({ existing, gpus, onSubmit, onCancel }: Props) {
  const { advanced } = useAdvancedMode()
  const [cameraId, setCameraId] = useState(existing?.camera_id ?? '')
  const [displayName, setDisplayName] = useState(existing?.display_name ?? '')
  const [streamUrl, setStreamUrl] = useState(existing ? '' : '')
  const [zoneId, setZoneId] = useState(existing?.zone_id ?? '')
  const [gpuId, setGpuId] = useState<string>(existing?.gpu_id != null ? String(existing.gpu_id) : '')
  const [resolutionW, setResolutionW] = useState(existing?.resolution_w ?? 1920)
  const [resolutionH, setResolutionH] = useState(existing?.resolution_h ?? 1080)
  const [fpsTarget, setFpsTarget] = useState(existing?.fps_target ?? 30)
  const [launchArgs, setLaunchArgs] = useState<CameraLaunchArgs>(existing?.launch_args ?? DEFAULT_LAUNCH_ARGS)
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  async function handleSubmit() {
    if (!cameraId || !displayName || (!existing && !streamUrl)) {
      setError('camera_id, display name, and stream URL are required.')
      return
    }
    setSubmitting(true)
    setError('')
    try {
      const input: CameraFormSubmit = {
        camera_id: cameraId,
        display_name: displayName,
        zone_id: zoneId || null,
        gpu_id: gpuId === '' ? null : Number(gpuId),
        resolution_w: resolutionW,
        resolution_h: resolutionH,
        fps_target: fpsTarget,
        launch_args: launchArgs,
      }
      // Only include stream_url if it's a create (always required) or the
      // operator actually typed a new one during an edit.
      if (!existing || streamUrl) {
        input.stream_url = streamUrl
      }
      await onSubmit(input)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save camera.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Modal
      title={existing ? `Edit ${existing.display_name}` : 'Add camera'}
      className="camera-form"
      onClose={onCancel}
    >
        {error && <div className="form-error">{error}</div>}

        <label>
          Camera ID
          <input
            value={cameraId}
            disabled={!!existing}
            placeholder="front_door"
            onChange={(e) => setCameraId(e.target.value)}
          />
        </label>
        <label>
          Display name
          <input
            value={displayName}
            placeholder="Front Door"
            onChange={(e) => setDisplayName(e.target.value)}
          />
        </label>
        <label>
          RTSP URL {existing && <span className="hint">(leave blank to keep current)</span>}
          <input
            value={streamUrl}
            placeholder={existing ? 'rtsp://••••:••••@host/...' : 'rtsp://user:pass@host/stream1'}
            onChange={(e) => setStreamUrl(e.target.value)}
          />
        </label>
        <label>
          Zone
          <input value={zoneId} placeholder="bedroom" onChange={(e) => setZoneId(e.target.value)} />
        </label>
        <label>
          GPU
          <select value={gpuId} onChange={(e) => setGpuId(e.target.value)}>
            <option value="">CPU / auto</option>
            {gpus.map((g) => (
              <option key={g.gpu_id} value={g.gpu_id}>
                GPU {g.gpu_id} — {g.name} ({g.memory_used_mb.toLocaleString()}/{g.memory_total_mb.toLocaleString()} MB used)
              </option>
            ))}
          </select>
        </label>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={launchArgs.recognize}
            onChange={(e) => setLaunchArgs({ ...launchArgs, recognize: e.target.checked })}
          />
          Recognize enrolled people
        </label>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={launchArgs.persist_events}
            onChange={(e) => setLaunchArgs({ ...launchArgs, persist_events: e.target.checked })}
          />
          Record entry/exit + alerts
        </label>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={launchArgs.enforce_liveness ?? false}
            onChange={(e) => setLaunchArgs({ ...launchArgs, enforce_liveness: e.target.checked })}
          />
          Enforce liveness (block photos/screens)
        </label>
        <label className="checkbox-label">
          <input
            type="checkbox"
            checked={launchArgs.collect_antispoofing ?? false}
            onChange={(e) => setLaunchArgs({ ...launchArgs, collect_antispoofing: e.target.checked })}
          />
          Collect anti-spoofing training data
        </label>

        {advanced && (
          <>
            <hr />
            <div className="advanced-section-label">Advanced</div>
            <div className="form-grid-2">
              <label>
                Resolution width
                <input type="number" value={resolutionW} onChange={(e) => setResolutionW(Number(e.target.value))} />
              </label>
              <label>
                Resolution height
                <input type="number" value={resolutionH} onChange={(e) => setResolutionH(Number(e.target.value))} />
              </label>
              <label>
                FPS target (metadata only)
                <input type="number" value={fpsTarget} onChange={(e) => setFpsTarget(Number(e.target.value))} />
              </label>
              <label>
                Processing FPS cap
                <input
                  type="number"
                  value={launchArgs.fps}
                  onChange={(e) => setLaunchArgs({ ...launchArgs, fps: Number(e.target.value) })}
                />
              </label>
              <label>
                Detection threshold (start track)
                <input
                  type="number" step="0.05" min={0.3} max={1}
                  value={launchArgs.det_thresh}
                  onChange={(e) => setLaunchArgs({ ...launchArgs, det_thresh: Number(e.target.value) })}
                />
              </label>
              <label>
                Re-ID threshold (restore old ID)
                <input
                  type="number" step="0.05" min={0.35} max={1}
                  value={launchArgs.reid_thresh}
                  onChange={(e) => setLaunchArgs({ ...launchArgs, reid_thresh: Number(e.target.value) })}
                />
              </label>
              <label>
                Min face size (px)
                <input
                  type="number"
                  value={launchArgs.min_face}
                  onChange={(e) => setLaunchArgs({ ...launchArgs, min_face: Number(e.target.value) })}
                />
              </label>
            </div>
            <label className="checkbox-label">
              <input
                type="checkbox"
                checked={launchArgs.cpu}
                onChange={(e) => setLaunchArgs({ ...launchArgs, cpu: e.target.checked })}
              />
              Force CPU inference (ignores GPU selection above)
            </label>
          </>
        )}

        <button className="btn-primary" disabled={submitting} onClick={handleSubmit}>
          {submitting ? 'Saving…' : existing ? 'Save changes' : 'Create camera'}
        </button>
    </Modal>
  )
}
