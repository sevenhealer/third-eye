import { useEffect, useRef } from 'react'
import { mjpegUrl } from '../api/client'

interface Props {
  cameraId: string
  displayName: string
  onClose: () => void
}

export function LiveViewModal({ cameraId, displayName, onClose }: Props) {
  const imgRef = useRef<HTMLImageElement>(null)

  useEffect(() => {
    const img = imgRef.current
    // Mirrors closeLiveView()'s explicit `img.src = ''` in the vanilla
    // dashboard: clearing src aborts the underlying request so the
    // server's pubsub.get_message() loop unsubscribes instead of
    // streaming to a closed modal forever.
    return () => {
      if (img) img.src = ''
    }
  }, [])

  return (
    <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal-box">
        <div className="modal-head">
          <strong>{displayName}</strong>
          <button onClick={onClose}>Close</button>
        </div>
        <img ref={imgRef} className="live-modal-img" src={mjpegUrl(cameraId)} alt={displayName} />
      </div>
    </div>
  )
}
