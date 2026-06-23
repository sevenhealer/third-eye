import { useState, useCallback } from 'react'
import { type CameraStatus, listCameras } from '../api/client'
import { usePolling } from '../api/usePolling'
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

  usePolling(refresh, 3000)

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
            // Pause every tile's MJPEG stream while the live modal is open:
            // the modal opens its own full-size stream, and browsers cap
            // concurrent connections per host (~6) - no point holding grid
            // streams nobody can see behind the overlay.
            <CameraTile
              key={cam.camera_id}
              camera={cam}
              paused={viewing !== undefined}
              onClick={() => setViewing(cam)}
            />
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
