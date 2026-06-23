import { useState, useCallback } from 'react'
import {
  type ZoneStatus,
  type PresenceLogEntry,
  ZONE_TYPES,
  listZones,
  createZone,
  updateZone,
  deleteZone,
  getPresenceLog,
} from '../api/client'
import { usePolling } from '../api/usePolling'
import { Modal } from '../components/Modal'
import { useFeedback } from '../components/feedbackContext'

export function ZonesPage() {
  const { toast, confirm } = useFeedback()
  const [zones, setZones] = useState<ZoneStatus[]>([])
  const [error, setError] = useState('')
  const [adding, setAdding] = useState(false)
  const [editing, setEditing] = useState<ZoneStatus | undefined>(undefined)
  const [viewingLog, setViewingLog] = useState<ZoneStatus | undefined>(undefined)
  const [logEntries, setLogEntries] = useState<PresenceLogEntry[]>([])
  const [logMessage, setLogMessage] = useState('')

  const refresh = useCallback(async () => {
    try {
      setZones(await listZones())
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load zones.')
    }
  }, [])

  usePolling(refresh, 3000)

  async function openLog(zone: ZoneStatus) {
    setViewingLog(zone)
    setLogEntries([])
    setLogMessage('Loading…')
    try {
      const entries = await getPresenceLog(zone.zone_id)
      setLogEntries(entries)
      setLogMessage(entries.length ? '' : 'No recorded visits yet.')
    } catch (e) {
      setLogMessage(e instanceof Error ? e.message : 'Failed to load log.')
    }
  }

  async function handleDelete(zone: ZoneStatus) {
    const ok = await confirm({
      title: 'Remove zone',
      message: `Remove "${zone.display_name}"? This only deletes the zone definition — `
        + `it's blocked if any camera is still assigned to it or it has recorded history.`,
      confirmLabel: 'Remove',
      danger: true,
    })
    if (!ok) return
    try {
      await deleteZone(zone.zone_id)
      await refresh()
      toast(`Zone "${zone.display_name}" removed.`, 'success')
    } catch (e) {
      toast(e instanceof Error ? e.message : 'Failed to remove zone.', 'error')
    }
  }

  return (
    <div>
      <div className="page-head">
        <h2>Zones</h2>
        <button className="btn-primary" onClick={() => setAdding(true)}>
          + Add zone
        </button>
      </div>
      {error && <div className="form-error">{error}</div>}

      <table className="data-table">
        <thead>
          <tr>
            <th>Zone</th>
            <th>Type</th>
            <th>Occupants</th>
            <th>Object counts</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {zones.map((z) => (
            <tr key={z.zone_id}>
              <td>
                <div>{z.display_name}</div>
                <div className="cell-sub">{z.zone_id}</div>
              </td>
              <td>
                <span className={`badge${z.zone_type === 'restricted' ? ' restricted' : ''}`}>
                  {z.zone_type}
                </span>
              </td>
              <td>{z.occupants.length ? z.occupants.map((o) => o.display_name).join(', ') : '—'}</td>
              <td>
                {Object.entries(z.object_counts)
                  .map(([k, v]) => `${k}:${v}`)
                  .join(', ') || '—'}
              </td>
              <td className="row-actions">
                <button onClick={() => setEditing(z)}>Edit</button>
                <button onClick={() => openLog(z)}>Log</button>
                <button className="btn-danger" onClick={() => handleDelete(z)}>
                  Remove
                </button>
              </td>
            </tr>
          ))}
          {!zones.length && (
            <tr>
              <td colSpan={5} className="empty">
                No zones yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>

      {adding && (
        <ZoneForm
          onClose={() => setAdding(false)}
          onSaved={async () => {
            setAdding(false)
            await refresh()
            toast('Zone created.', 'success')
          }}
        />
      )}

      {editing && (
        <ZoneForm
          existing={editing}
          onClose={() => setEditing(undefined)}
          onSaved={async () => {
            setEditing(undefined)
            await refresh()
            toast('Zone updated.', 'success')
          }}
        />
      )}

      {viewingLog && (
        <Modal
          title={`${viewingLog.display_name} — who came and went`}
          onClose={() => setViewingLog(undefined)}
        >
          {logMessage && <div className="hint">{logMessage}</div>}
          <table className="data-table">
            <thead>
              <tr>
                <th>Who</th>
                <th>Entered</th>
                <th>Exited</th>
              </tr>
            </thead>
            <tbody>
              {logEntries.map((e, i) => (
                <tr key={i} className={`log-row ${e.is_unknown ? 'unknown' : ''}`}>
                  <td>{e.display_name}</td>
                  <td>{new Date(e.entry_time).toLocaleString()}</td>
                  <td>{e.exit_time ? new Date(e.exit_time).toLocaleString() : <span className="still-here">still here</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Modal>
      )}
    </div>
  )
}

function ZoneForm({
  existing,
  onClose,
  onSaved,
}: {
  existing?: ZoneStatus
  onClose: () => void
  onSaved: () => void
}) {
  const [zoneId, setZoneId] = useState(existing?.zone_id ?? '')
  const [displayName, setDisplayName] = useState(existing?.display_name ?? '')
  const [zoneType, setZoneType] = useState<string>(existing?.zone_type ?? 'general')
  const [description, setDescription] = useState(existing?.description ?? '')
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)

  async function submit() {
    if (!existing && !zoneId.trim()) {
      setError('Zone ID is required.')
      return
    }
    if (!displayName.trim()) {
      setError('Display name is required.')
      return
    }
    setSubmitting(true)
    setError('')
    try {
      if (existing) {
        await updateZone(existing.zone_id, {
          display_name: displayName.trim(),
          zone_type: zoneType,
          description: description.trim() || undefined,
        })
      } else {
        await createZone({
          zone_id: zoneId.trim(),
          display_name: displayName.trim(),
          zone_type: zoneType,
          description: description.trim() || undefined,
        })
      }
      onSaved()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save zone.')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Modal
      title={existing ? `Edit ${existing.display_name}` : 'Add zone'}
      className="camera-form"
      width={320}
      onClose={onClose}
    >
      {error && <div className="form-error">{error}</div>}
      {!existing && (
        <label>
          Zone ID
          <input value={zoneId} placeholder="dining_room" onChange={(e) => setZoneId(e.target.value)} />
        </label>
      )}
      <label>
        Display name
        <input value={displayName} placeholder="Dining Room" onChange={(e) => setDisplayName(e.target.value)} />
      </label>
      <label>
        Type
        <select value={zoneType} onChange={(e) => setZoneType(e.target.value)}>
          {ZONE_TYPES.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </label>
      <label>
        Description <span className="hint">(optional)</span>
        <input
          value={description}
          placeholder="e.g. main floor dining area"
          onChange={(e) => setDescription(e.target.value)}
        />
      </label>
      <button className="btn-primary" disabled={submitting} onClick={submit}>
        {submitting ? 'Saving…' : existing ? 'Save changes' : 'Create zone'}
      </button>
    </Modal>
  )
}
