import { useCallback, useEffect, useState } from 'react'
import {
  type AntiSpoofDataset,
  type AntiSpoofItem,
  type TrainStatus,
  type CollectionCamera,
  getAntiSpoofDataset,
  relabelCrop,
  deleteCrop,
  startTraining,
  getTrainStatus,
  listCollection,
  setCollection,
  fetchCropBlobUrl,
} from '../api/client'
import { usePolling } from '../api/usePolling'
import { useFeedback } from '../components/feedbackContext'

// url -> blob URL, fetched once per page session (an <img src> can't carry the
// auth header). Keyed by url, which encodes the label folder, so a relabel
// (which changes the url) re-fetches into the new label.
const cropCache = new Map<string, string>()

function CropThumb({ item }: { item: AntiSpoofItem }) {
  const [url, setUrl] = useState<string | undefined>(cropCache.get(item.url))
  useEffect(() => {
    if (cropCache.has(item.url)) return
    let cancelled = false
    fetchCropBlobUrl(item.url).then((b) => {
      if (cancelled || !b) return
      cropCache.set(item.url, b)
      setUrl(b)
    })
    return () => {
      cancelled = true
    }
  }, [item.url])
  return (
    <div className={`as-thumb as-thumb-${item.label}`}>
      {url ? <img src={url} alt={item.filename} /> : <div className="as-thumb-loading" />}
    </div>
  )
}

export function AntiSpoofingPage() {
  const { toast, confirm } = useFeedback()
  const [data, setData] = useState<AntiSpoofDataset | null>(null)
  const [train, setTrain] = useState<TrainStatus | null>(null)
  const [cameras, setCameras] = useState<CollectionCamera[]>([])
  const [epochs, setEpochs] = useState(30)
  const [error, setError] = useState('')

  const refresh = useCallback(async () => {
    try {
      const [ds, ts, cams] = await Promise.all([
        getAntiSpoofDataset(),
        getTrainStatus(),
        listCollection(),
      ])
      setData(ds)
      setTrain(ts)
      setCameras(cams)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load anti-spoofing data.')
    }
  }, [])

  const toggleCollection = async (cam: CollectionCamera) => {
    try {
      await setCollection(cam.camera_id, !cam.collecting)
      toast(
        `Collection ${!cam.collecting ? 'started' : 'stopped'} on ${cam.display_name}`,
        'success',
      )
      refresh()
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Could not change collection', 'error')
    }
  }

  usePolling(refresh, 4000)

  const onRelabel = async (item: AntiSpoofItem) => {
    const target = item.label === 'live' ? 'spoof' : 'live'
    try {
      await relabelCrop(item.label, item.filename, target)
      toast(`Re-labelled as ${target}`, 'success')
      refresh()
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Relabel failed', 'error')
    }
  }

  const onDelete = async (item: AntiSpoofItem) => {
    try {
      await deleteCrop(item.label, item.filename)
      refresh()
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Delete failed', 'error')
    }
  }

  const onTrain = async () => {
    const ok = await confirm({
      title: 'Train CDCN++ now?',
      message: `Trains on the current dataset (live=${data?.live ?? 0}, spoof=${data?.spoof ?? 0}) for ${epochs} epochs on the box GPU. Runs in the background; progress shows here.`,
      confirmLabel: 'Start training',
    })
    if (!ok) return
    try {
      const ts = await startTraining(epochs)
      setTrain(ts)
      toast('Training started', 'success')
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Could not start training', 'error')
    }
  }

  const running = train?.running ?? false
  const pct =
    train?.latest_epoch && train?.total_epochs
      ? Math.round((train.latest_epoch / train.total_epochs) * 100)
      : 0

  return (
    <>
      <div className="page-head">
        <h2>Anti-Spoofing</h2>
      </div>
      <p className="hint">
        Face crops are auto-collected from cameras with “Collect anti-spoofing
        training data” on, auto-labelled <strong>live</strong> or{' '}
        <strong>spoof</strong> by the liveness gate. Correct any wrong labels
        below, then train the CDCN++ texture model.
      </p>

      {error && <div className="form-error">{error}</div>}

      <div className="as-collection">
        <span className="as-collection-title">Data collection</span>
        {cameras.length === 0 && <span className="hint">No active cameras.</span>}
        {cameras.map((cam) => (
          <div key={cam.camera_id} className="as-collection-cam">
            <span className={`as-dot ${cam.collecting ? 'on' : ''}`} />
            <span className="as-collection-name">{cam.display_name}</span>
            <button
              className={cam.collecting ? 'btn-danger' : 'btn-primary'}
              onClick={() => toggleCollection(cam)}
            >
              {cam.collecting ? 'Stop' : 'Start'}
            </button>
          </div>
        ))}
      </div>

      <div className="as-stats">
        <div className="as-stat as-stat-live">
          <span className="as-stat-num">{data?.live ?? '—'}</span>
          <span className="as-stat-label">live crops</span>
        </div>
        <div className="as-stat as-stat-spoof">
          <span className="as-stat-num">{data?.spoof ?? '—'}</span>
          <span className="as-stat-label">spoof crops</span>
        </div>
        <div className="as-train-control">
          <label>
            Epochs
            <input
              type="number"
              min={1}
              max={500}
              value={epochs}
              disabled={running}
              onChange={(e) => setEpochs(Number(e.target.value))}
            />
          </label>
          <button className="btn-primary" onClick={onTrain} disabled={running}>
            {running ? 'Training…' : 'Train CDCN++'}
          </button>
        </div>
      </div>

      {(running || train?.finished) && (
        <div className="as-train-status">
          <div className="as-train-row">
            <span>
              {running
                ? `Training — epoch ${train?.latest_epoch ?? 0}/${train?.total_epochs ?? epochs}`
                : 'Last run finished'}
            </span>
            {train?.best_val_acc != null && (
              <span>best val_acc {train.best_val_acc.toFixed(3)}</span>
            )}
          </div>
          <div className="as-progress">
            <div className="as-progress-fill" style={{ width: `${pct}%` }} />
          </div>
          {train?.tail && train.tail.length > 0 && (
            <pre className="as-train-log">{train.tail.join('\n')}</pre>
          )}
        </div>
      )}

      <div className="as-grid">
        {data?.items.map((item) => (
          <div key={`${item.label}/${item.filename}`} className="as-card">
            <CropThumb item={item} />
            <div className="as-card-actions">
              <span className={`as-badge as-badge-${item.label}`}>{item.label}</span>
              <button className="link-button" onClick={() => onRelabel(item)}>
                → {item.label === 'live' ? 'spoof' : 'live'}
              </button>
              <button className="link-button as-del" onClick={() => onDelete(item)}>
                ✕
              </button>
            </div>
          </div>
        ))}
      </div>

      {data && data.items.length === 0 && (
        <div className="empty">
          No crops yet. Turn on “Collect anti-spoofing training data” for a camera
          (Cameras → edit), let it run, then come back to review the auto-labels.
        </div>
      )}
    </>
  )
}
