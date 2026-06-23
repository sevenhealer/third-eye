import { useEffect, useState, useCallback } from 'react'
import {
  type CameraStatus,
  type CameraDetail,
  type GpuStat,
  type NewCameraInput,
  listCameras,
  getCamera,
  createCamera,
  updateCamera,
  startCamera,
  stopCamera,
  deleteCamera,
  listGpus,
} from '../api/client'
import { CameraForm, type CameraFormSubmit } from '../components/CameraForm'
import { useAdvancedMode } from '../components/advancedModeContext'

function statusDotClass(cam: CameraStatus): string {
  if (cam.process_state === 'running') return 'dot-online'
  if (cam.process_state === 'crashed') return 'dot-error'
  return 'dot-offline'
}

export function CamerasPage() {
  const { advanced } = useAdvancedMode()
  const [cameras, setCameras] = useState<CameraStatus[]>([])
  const [gpus, setGpus] = useState<GpuStat[]>([])
  const [editing, setEditing] = useState<CameraDetail | undefined>(undefined)
  const [adding, setAdding] = useState(false)
  const [error, setError] = useState('')

  const refresh = useCallback(async () => {
    try {
      setCameras(await listCameras())
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load cameras.')
    }
  }, [])

  useEffect(() => {
    // refresh()'s setState calls happen after its internal await, not
    // synchronously within this effect body - standard fetch-on-mount,
    // not the derived-state-from-props anti-pattern this lint rule targets.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refresh()
    listGpus().then(setGpus).catch(() => setGpus([]))
    const interval = setInterval(refresh, 3000)
    return () => clearInterval(interval)
  }, [refresh])

  async function handleCreate(input: CameraFormSubmit) {
    await createCamera(input as NewCameraInput)
    setAdding(false)
    await refresh()
  }

  async function handleSaveEdit(input: CameraFormSubmit) {
    if (!editing) return
    const { camera_id, ...patch } = input
    void camera_id // never sent - the backend takes it from the URL path and forbids it in the body
    await updateCamera(editing.camera_id, patch)
    setEditing(undefined)
    await refresh()
  }

  async function openEdit(cameraId: string) {
    setError('')
    try {
      setEditing(await getCamera(cameraId))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load camera.')
    }
  }

  async function handleStart(cameraId: string) {
    setError('')
    try {
      await startCamera(cameraId)
      await refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to start camera.')
    }
  }

  async function handleStop(cameraId: string) {
    setError('')
    try {
      await stopCamera(cameraId)
      await refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to stop camera.')
    }
  }

  async function handleDelete(cameraId: string) {
    if (!confirm(`Remove ${cameraId}? Its history is kept, it just stops showing up here.`)) return
    setError('')
    try {
      await deleteCamera(cameraId)
      await refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to delete camera.')
    }
  }

  return (
    <div>
      <div className="page-head">
        <h2>Cameras</h2>
        <button className="btn-primary" onClick={() => setAdding(true)}>
          + Add camera
        </button>
      </div>
      {error && <div className="form-error">{error}</div>}

      <table className="data-table">
        <thead>
          <tr>
            <th></th>
            <th>Name</th>
            <th>Zone</th>
            {advanced && <th>GPU</th>}
            <th>Status</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {cameras.map((cam) => (
            <tr key={cam.camera_id}>
              <td>
                <span className={`status-dot ${statusDotClass(cam)}`} />
              </td>
              <td>{cam.display_name}</td>
              <td>{cam.zone_id || '—'}</td>
              {advanced && <td>{cam.gpu_id != null ? `GPU ${cam.gpu_id}` : 'CPU/auto'}</td>}
              <td>
                {cam.process_state || cam.desired_state}
                {cam.process_state === 'crashed' && ' ⚠'}
              </td>
              <td className="row-actions">
                {cam.desired_state === 'running' ? (
                  <button onClick={() => handleStop(cam.camera_id)}>Stop</button>
                ) : (
                  <button onClick={() => handleStart(cam.camera_id)}>Start</button>
                )}
                <button onClick={() => openEdit(cam.camera_id)}>Edit</button>
                <button className="btn-danger" onClick={() => handleDelete(cam.camera_id)}>
                  Remove
                </button>
              </td>
            </tr>
          ))}
          {!cameras.length && (
            <tr>
              <td colSpan={6} className="empty">
                No cameras yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>

      {adding && <CameraForm gpus={gpus} onSubmit={handleCreate} onCancel={() => setAdding(false)} />}
      {editing && (
        <CameraForm existing={editing} gpus={gpus} onSubmit={handleSaveEdit} onCancel={() => setEditing(undefined)} />
      )}
    </div>
  )
}
