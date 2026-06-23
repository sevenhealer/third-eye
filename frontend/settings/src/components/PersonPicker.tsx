import { useEffect, useState } from 'react'
import { type Person } from '../api/client'
import { getAllPersons } from '../api/personsCache'
import { Modal } from './Modal'

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
    <Modal title={title} width={320} className="person-picker" onClose={onClose}>
      <input
        className="picker-search"
        placeholder="Search people..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <div className="picker-list">
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
    </Modal>
  )
}
