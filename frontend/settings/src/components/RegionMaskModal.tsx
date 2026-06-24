import { useRef, useState } from 'react'
import type { CameraDetail } from '../api/client'
import { mjpegUrl, updateCamera } from '../api/client'
import { Modal } from './Modal'

type Rect = [number, number, number, number] // normalized x1,y1,x2,y2

interface Props {
  camera: CameraDetail
  onClose: () => void
  onSaved: () => void
}

// Draw rectangles over the live feed to mask regions whose detections should be
// dropped (a ticking burnt-in timestamp, a fixed false positive like a statue).
// Coordinates are normalized 0-1 so they survive any resolution change.
export function RegionMaskModal({ camera, onClose, onSaved }: Props) {
  const [rects, setRects] = useState<Rect[]>(
    (camera.launch_args.ignore_regions as Rect[] | undefined) ?? [],
  )
  const [drag, setDrag] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const boxRef = useRef<HTMLDivElement>(null)

  const toNorm = (e: React.MouseEvent) => {
    const r = boxRef.current!.getBoundingClientRect()
    return {
      x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
      y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
    }
  }

  const onDown = (e: React.MouseEvent) => {
    const p = toNorm(e)
    setDrag({ x0: p.x, y0: p.y, x1: p.x, y1: p.y })
  }
  const onMove = (e: React.MouseEvent) => {
    if (!drag) return
    const p = toNorm(e)
    setDrag({ ...drag, x1: p.x, y1: p.y })
  }
  const onUp = () => {
    if (!drag) return
    const x1 = Math.min(drag.x0, drag.x1)
    const y1 = Math.min(drag.y0, drag.y1)
    const x2 = Math.max(drag.x0, drag.x1)
    const y2 = Math.max(drag.y0, drag.y1)
    if (x2 - x1 > 0.01 && y2 - y1 > 0.01) {
      setRects([...rects, [+x1.toFixed(4), +y1.toFixed(4), +x2.toFixed(4), +y2.toFixed(4)]])
    }
    setDrag(null)
  }

  const save = async () => {
    setSaving(true)
    setError('')
    try {
      // Send the full launch_args (merged) so other fields are preserved.
      await updateCamera(camera.camera_id, {
        launch_args: { ...camera.launch_args, ignore_regions: rects },
      })
      onSaved()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Save failed')
      setSaving(false)
    }
  }

  const pct = (v: number) => `${v * 100}%`
  const live =
    drag && {
      left: pct(Math.min(drag.x0, drag.x1)),
      top: pct(Math.min(drag.y0, drag.y1)),
      width: pct(Math.abs(drag.x1 - drag.x0)),
      height: pct(Math.abs(drag.y1 - drag.y0)),
    }

  return (
    <Modal title={`Mask regions — ${camera.display_name}`} onClose={onClose}>
      <p className="hint">
        Drag a box over anything that should be ignored — the burnt-in timestamp
        (its ticking digits fake motion) or a fixed false positive like a statue.
        Detections centered inside a box are dropped.
      </p>
      <div
        ref={boxRef}
        className="mask-canvas"
        onMouseDown={onDown}
        onMouseMove={onMove}
        onMouseUp={onUp}
        onMouseLeave={onUp}
      >
        <img src={mjpegUrl(camera.camera_id)} alt="camera feed" draggable={false} />
        {rects.map((r, i) => (
          <div
            key={i}
            className="mask-rect"
            style={{ left: pct(r[0]), top: pct(r[1]), width: pct(r[2] - r[0]), height: pct(r[3] - r[1]) }}
          >
            <button
              className="mask-rect-del"
              onClick={(e) => {
                e.stopPropagation()
                setRects(rects.filter((_, j) => j !== i))
              }}
            >
              ✕
            </button>
          </div>
        ))}
        {live && <div className="mask-rect mask-rect-live" style={live} />}
      </div>
      {error && <div className="form-error">{error}</div>}
      <div className="modal-actions">
        <span className="hint">{rects.length} mask(s)</span>
        <button onClick={() => setRects([])} disabled={!rects.length}>
          Clear all
        </button>
        <button className="btn-primary" onClick={save} disabled={saving}>
          {saving ? 'Saving…' : 'Save & apply'}
        </button>
      </div>
    </Modal>
  )
}
