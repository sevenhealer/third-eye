import type { CameraStatus } from '../api/client'
import { mjpegUrl } from '../api/client'

function dotClass(status: string): string {
  if (status === 'online') return 'dot-online'
  return 'dot-offline'
}

export function CameraTile({
  camera,
  onClick,
  paused = false,
}: {
  camera: CameraStatus
  onClick: () => void
  paused?: boolean
}) {
  // src is computed only from camera_id (+ the session token, stable for
  // the page's lifetime) and the explicit `paused` flag - never from
  // anything that changes on the 3s poll tick. React only touches the DOM
  // <img src> when this *value* changes, so a re-render from polling
  // (status/name updating) doesn't reset/restart the MJPEG stream - same
  // invariant the vanilla dashboard's cameraTiles cache protected. An
  // empty src while paused drops the connection (the modal owns the feed).
  return (
    <div className="cam-tile" onClick={onClick}>
      <img src={paused ? '' : mjpegUrl(camera.camera_id)} alt={camera.display_name} />
      <div className="label">
        <span>
          <span className={`status-dot ${dotClass(camera.status)}`} /> {camera.display_name}
        </span>
        <span>{camera.zone_id || ''}</span>
      </div>
    </div>
  )
}
