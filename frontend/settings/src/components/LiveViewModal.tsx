import { useEffect, useRef } from 'react'
import { mjpegUrl } from '../api/client'
import { Modal } from './Modal'

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
    <Modal title={displayName} onClose={onClose}>
      <img ref={imgRef} className="live-modal-img" src={mjpegUrl(cameraId)} alt={displayName} />
    </Modal>
  )
}
