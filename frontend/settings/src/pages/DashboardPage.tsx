import { useEffect, useState, useCallback } from 'react'
import { type CameraStatus, listCameras } from '../api/client'
import { CameraTile } from '../components/CameraTile'
import { LiveViewModal } from '../components/LiveViewModal'

export function DashboardPage() {
  const [cameras, setCameras] = useState<CameraStatus[]>([])
  const [error, setError] = useState('')
  const [viewing, setViewing] = useState<CameraStatus | undefined>(undefined)

  const refresh = useCallback(async () => {
    try {
      setCameras(await listCameras())
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load cameras.')
    }
  }, [])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refresh()
    const interval = setInterval(refresh, 3000)
    return () => clearInterval(interval)
  }, [refresh])

  return (
    <div>
      <div className="page-head">
        <h2>Cameras</h2>
      </div>
      {error && <div className="form-error">{error}</div>}

      {!cameras.length ? (
        <div className="empty">No cameras registered. Add one under Cameras.</div>
      ) : (
        <div className="cam-grid">
          {cameras.map((cam) => (
            <CameraTile key={cam.camera_id} camera={cam} onClick={() => setViewing(cam)} />
          ))}
        </div>
      )}

      {viewing && (
        <LiveViewModal
          cameraId={viewing.camera_id}
          displayName={viewing.display_name}
          onClose={() => setViewing(undefined)}
        />
      )}
    </div>
  )
}
