import { useState, useCallback } from 'react'
import {
  type ZoneStatus,
  type PresenceLogEntry,
  ZONE_TYPES,
  listZones,
  createZone,
  updateZoneType,
  getPresenceLog,
} from '../api/client'
import { usePolling } from '../api/usePolling'
import { Modal } from '../components/Modal'
import { useFeedback } from '../components/feedbackContext'

export function ZonesPage() {
  const { toast } = useFeedback()
  const [zones, setZones] = useState<ZoneStatus[]>([])
  const [error, setError] = useState('')
  const [adding, setAdding] = useState(false)
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

  async function handleTypeChange(zoneId: string, newType: string) {
    try {
      await updateZoneType(zoneId, newType)
      await refresh()
      toast('Zone type updated.', 'success')
    } catch (e) {
      toast(`Couldn't update zone type: ${e instanceof Error ? e.message : 'unknown error'}`, 'error')
    }
  }

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
          </tr>
        </thead>
        <tbody>
          {zones.map((z) => (
            <tr key={z.zone_id} className="clickable" onClick={() => openLog(z)}>
              <td>{z.display_name}</td>
              <td>
                <select
                  className="zone-type-select"
                  value={z.zone_type}
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => handleTypeChange(z.zone_id, e.target.value)}
                >
                  {ZONE_TYPES.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              </td>
              <td>{z.occupants.length ? z.occupants.map((o) => o.display_name).join(', ') : '—'}</td>
              <td>
                {Object.entries(z.object_counts)
                  .map(([k, v]) => `${k}:${v}`)
                  .join(', ') || '—'}
              </td>
            </tr>
          ))}
          {!zones.length && (
            <tr>
              <td colSpan={4} className="empty">
                No zones yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>

      {adding && (
        <AddZoneModal
          onClose={() => setAdding(false)}
          onCreated={async () => {
            setAdding(false)
            await refresh()
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

function AddZoneModal({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const [zoneId, setZoneId] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [zoneType, setZoneType] = useState<string>('general')
  const [error, setError] = useState('')

  async function submit() {
    if (!zoneId.trim() || !displayName.trim()) {
      setError('zone_id and display name are both required.')
      return
    }
    try {
      await createZone({ zone_id: zoneId.trim(), display_name: displayName.trim(), zone_type: zoneType })
      onCreated()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to create zone.')
    }
  }

  return (
    <Modal title="Add zone" className="camera-form" width={300} onClose={onClose}>
      {error && <div className="form-error">{error}</div>}
      <label>
        Zone ID
        <input value={zoneId} placeholder="dining_room" onChange={(e) => setZoneId(e.target.value)} />
      </label>
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
      <button className="btn-primary" onClick={submit}>
        Create zone
      </button>
    </Modal>
  )
}
