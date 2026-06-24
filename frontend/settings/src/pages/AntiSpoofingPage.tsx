import { useCallback, useEffect, useState } from 'react'
import {
  type AntiSpoofDataset,
  type AntiSpoofItem,
  type TrainStatus,
  type CollectionCamera,
  type Snapshot,
  type Checkpoint,
  getAntiSpoofDataset,
  relabelCrop,
  deleteCrop,
  clearDataset,
  startTraining,
  getTrainStatus,
  listCollection,
  setCollection,
  fetchCropBlobUrl,
  listSnapshots,
  saveSnapshot,
  restoreSnapshot,
  deleteSnapshot,
  listCheckpoints,
  saveCheckpoint,
  activateCheckpoint,
  deleteCheckpoint,
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
  const [snapshots, setSnapshots] = useState<Snapshot[]>([])
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [snapName, setSnapName] = useState('')
  const [ckptName, setCkptName] = useState('')

  const refresh = useCallback(async () => {
    try {
      const [ds, ts, cams, snaps, ckpts] = await Promise.all([
        getAntiSpoofDataset(),
        getTrainStatus(),
        listCollection(),
        listSnapshots(),
        listCheckpoints(),
      ])
      setData(ds)
      setTrain(ts)
      setCameras(cams)
      setSnapshots(snaps)
      setCheckpoints(ckpts)
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

  const onClear = async (label?: 'live' | 'spoof') => {
    const what = label ? `all ${label} crops` : 'the ENTIRE dataset'
    const ok = await confirm({
      title: `Delete ${what}?`,
      message: `This permanently removes ${what}. This cannot be undone.`,
      confirmLabel: 'Delete',
    })
    if (!ok) return
    try {
      const res = await clearDataset(label)
      toast(`Deleted ${res.deleted} crop(s)`, 'success')
      refresh()
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Clear failed', 'error')
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
      toast('Training started — the model you had is auto-archived first', 'success')
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Could not start training', 'error')
    }
  }

  const onSaveSnapshot = async () => {
    const name = snapName.trim()
    if (!name) return
    try {
      await saveSnapshot(name)
      setSnapName('')
      toast(`Snapshot “${name}” saved`, 'success')
      refresh()
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Snapshot failed', 'error')
    }
  }

  const onRestoreSnapshot = async (s: Snapshot) => {
    const ok = await confirm({
      title: `Restore “${s.name}”?`,
      message: `This REPLACES the working dataset (live=${data?.live ?? 0}, spoof=${data?.spoof ?? 0}) with the snapshot (live=${s.live}, spoof=${s.spoof}). Current crops are cleared first.`,
      confirmLabel: 'Restore',
    })
    if (!ok) return
    try {
      const res = await restoreSnapshot(s.name)
      toast(`Restored ${res.restored} crop(s) from “${s.name}”`, 'success')
      refresh()
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Restore failed', 'error')
    }
  }

  const onDeleteSnapshot = async (s: Snapshot) => {
    const ok = await confirm({
      title: `Delete snapshot “${s.name}”?`,
      message: 'Permanently removes the saved snapshot. The working dataset is untouched.',
      confirmLabel: 'Delete',
    })
    if (!ok) return
    try {
      await deleteSnapshot(s.name)
      refresh()
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Delete failed', 'error')
    }
  }

  const onSaveCheckpoint = async () => {
    const name = ckptName.trim()
    if (!name) return
    try {
      await saveCheckpoint(name)
      setCkptName('')
      toast(`Checkpoint “${name}” saved`, 'success')
      refresh()
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Save failed', 'error')
    }
  }

  const onActivateCheckpoint = async (c: Checkpoint) => {
    const ok = await confirm({
      title: `Activate “${c.name}”?`,
      message: `Makes this the live model${c.val_acc != null ? ` (val_acc ${c.val_acc.toFixed(3)})` : ''}. Restart the camera(s) afterwards to load it.`,
      confirmLabel: 'Activate',
    })
    if (!ok) return
    try {
      const res = await activateCheckpoint(c.name)
      toast(res.note || 'Checkpoint activated', 'success')
      refresh()
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Activate failed', 'error')
    }
  }

  const onDeleteCheckpoint = async (c: Checkpoint) => {
    const ok = await confirm({
      title: `Delete checkpoint “${c.name}”?`,
      message: 'Permanently removes the archived checkpoint.',
      confirmLabel: 'Delete',
    })
    if (!ok) return
    try {
      await deleteCheckpoint(c.name)
      refresh()
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Delete failed', 'error')
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
          {!!data?.live && (
            <button className="link-button as-del" onClick={() => onClear('live')}>
              clear
            </button>
          )}
        </div>
        <div className="as-stat as-stat-spoof">
          <span className="as-stat-num">{data?.spoof ?? '—'}</span>
          <span className="as-stat-label">spoof crops</span>
          {!!data?.spoof && (
            <button className="link-button as-del" onClick={() => onClear('spoof')}>
              clear
            </button>
          )}
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

      <div className="as-manage">
        <div className="as-panel">
          <div className="as-panel-head">
            <h3>Dataset snapshots</h3>
            <div className="as-panel-add">
              <input
                placeholder="snapshot name"
                value={snapName}
                onChange={(e) => setSnapName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && onSaveSnapshot()}
              />
              <button className="btn-primary" onClick={onSaveSnapshot} disabled={!snapName.trim()}>
                Save current
              </button>
            </div>
          </div>
          <p className="hint">
            Save the current labelled set, then Clear all and collect a fresh one.
            Restore a snapshot to train on it again.
          </p>
          {snapshots.length === 0 ? (
            <div className="empty">No snapshots yet.</div>
          ) : (
            <ul className="as-list">
              {snapshots.map((s) => (
                <li key={s.name} className="as-list-row">
                  <span className="as-list-name">{s.name}</span>
                  <span className="as-list-meta">
                    live {s.live} · spoof {s.spoof}
                  </span>
                  <button onClick={() => onRestoreSnapshot(s)}>Restore</button>
                  <button className="link-button as-del" onClick={() => onDeleteSnapshot(s)}>
                    ✕
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="as-panel">
          <div className="as-panel-head">
            <h3>Model checkpoints</h3>
            <div className="as-panel-add">
              <input
                placeholder="checkpoint name"
                value={ckptName}
                onChange={(e) => setCkptName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && onSaveCheckpoint()}
              />
              <button className="btn-primary" onClick={onSaveCheckpoint} disabled={!ckptName.trim()}>
                Save active
              </button>
            </div>
          </div>
          <p className="hint">
            Training overwrites the active model from scratch (the previous one is
            auto-archived). Re-activate an older one if a retrain came out worse.
          </p>
          {checkpoints.length === 0 ? (
            <div className="empty">No model yet — train one first.</div>
          ) : (
            <ul className="as-list">
              {checkpoints.map((c) => (
                <li key={c.name} className={`as-list-row${c.active ? ' as-active' : ''}`}>
                  <span className="as-list-name">
                    {c.name}
                    {c.active && <span className="as-active-tag">active</span>}
                  </span>
                  <span className="as-list-meta">
                    {c.val_acc != null ? `val_acc ${c.val_acc.toFixed(3)}` : 'acc —'}
                    {c.epoch != null ? ` · ep ${c.epoch}` : ''}
                  </span>
                  {!c.name.startsWith('(') && !c.active && (
                    <button onClick={() => onActivateCheckpoint(c)}>Activate</button>
                  )}
                  {!c.name.startsWith('(') && (
                    <button className="link-button as-del" onClick={() => onDeleteCheckpoint(c)}>
                      ✕
                    </button>
                  )}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {!!(data && (data.live || data.spoof)) && (
        <div className="as-grid-head">
          <span className="hint">
            Showing {data.items.length} most recent of {data.live + data.spoof} crop(s)
          </span>
          <button className="btn-danger" onClick={() => onClear()}>
            Clear all
          </button>
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
