import type { CameraStatus } from '../api/client'
import { mjpegUrl } from '../api/client'

function dotClass(status: string): string {
  if (status === 'online') return 'dot-online'
  return 'dot-offline'
}

export function CameraTile({ camera, onClick }: { camera: CameraStatus; onClick: () => void }) {
  // src is computed only from camera_id (+ the session token, stable for
  // the page's lifetime) - never from anything that changes on the 3s
  // poll tick. React only touches the DOM <img src> when this *value*
  // changes, so a re-render from polling (status/name updating) doesn't
  // reset/restart the MJPEG stream - same invariant the vanilla
  // dashboard's cameraTiles cache protected, just expressed as "don't
  // derive src from volatile state" instead of "build the DOM node once."
  return (
    <div className="cam-tile" onClick={onClick}>
      <img src={mjpegUrl(camera.camera_id)} alt={camera.display_name} />
      <div className="label">
        <span>
          <span className={`status-dot ${dotClass(camera.status)}`} /> {camera.display_name}
        </span>
        <span>{camera.zone_id || ''}</span>
      </div>
    </div>
  )
}
