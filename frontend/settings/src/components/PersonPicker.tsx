import { useEffect, useState } from 'react'
import { type Person } from '../api/client'
import { getAllPersons } from '../api/personsCache'

export function PersonPickerModal({
  title = 'Merge into…',
  onClose,
  onPick,
}: {
  title?: string
  onClose: () => void
  onPick: (personId: string, name: string) => void
}) {
  const [persons, setPersons] = useState<Person[]>([])
  const [query, setQuery] = useState('')

  useEffect(() => {
    getAllPersons().then(setPersons)
  }, [])

  const matches = persons.filter((p) => p.display_name.toLowerCase().includes(query.trim().toLowerCase()))

  return (
    <div className="modal-overlay" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal-box" style={{ width: 320, maxHeight: '70vh', display: 'flex', flexDirection: 'column' }}>
        <div className="modal-head">
          <strong>{title}</strong>
          <button onClick={onClose}>Close</button>
        </div>
        <input
          autoFocus
          placeholder="Search people..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          style={{
            marginBottom: 8,
            padding: 7,
            background: '#0e1116',
            color: '#e6e6e6',
            border: '1px solid #2a2f37',
            borderRadius: 4,
            boxSizing: 'border-box',
          }}
        />
        <div style={{ overflowY: 'auto', flex: 1 }}>
          {matches.length ? (
            matches.map((p) => (
              <div key={p.person_id} className="merge-row" onClick={() => onPick(p.person_id, p.display_name)}>
                {p.display_name}
              </div>
            ))
          ) : (
            <div className="empty">No match.</div>
          )}
        </div>
      </div>
    </div>
  )
}
