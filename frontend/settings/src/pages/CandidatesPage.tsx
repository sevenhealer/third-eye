import { useEffect, useState, useCallback } from 'react'
import {
  type Candidate,
  type Person,
  listCandidates,
  approveCandidate,
  mergeCandidate,
  rejectCandidate,
  listIdentities,
  fetchCropBlobUrl,
} from '../api/client'

// candidate_id -> blob URL, fetched once per page session (mirrors the
// vanilla dashboard's candImageCache) - module-level so it survives
// across the 4s poll's re-renders without re-fetching.
const cropUrlCache = new Map<string, string>()

// All enrolled persons, fetched lazily the first time a merge picker
// opens and cached until a new person is approved (mirrors vanilla's
// mergeAllPersons).
let allPersonsCache: Person[] | null = null
function invalidatePersonsCache() {
  allPersonsCache = null
}
async function getAllPersons(): Promise<Person[]> {
  if (allPersonsCache === null) {
    allPersonsCache = await listIdentities()
  }
  return allPersonsCache
}

function useCandidateCropUrl(candidateId: string, cropUrl: string | undefined): string | undefined {
  const [url, setUrl] = useState<string | undefined>(cropUrlCache.get(candidateId))

  useEffect(() => {
    if (!cropUrl || cropUrlCache.has(candidateId)) return
    let cancelled = false
    fetchCropBlobUrl(cropUrl).then((blobUrl) => {
      if (cancelled || !blobUrl) return
      cropUrlCache.set(candidateId, blobUrl)
      setUrl(blobUrl)
    })
    return () => {
      cancelled = true
    }
  }, [candidateId, cropUrl])

  return url
}

export function CandidatesPage() {
  const [candidates, setCandidates] = useState<Candidate[]>([])
  const [error, setError] = useState('')

  const refresh = useCallback(async () => {
    try {
      const data = await listCandidates()
      setCandidates(data.candidates)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load candidates.')
    }
  }, [])

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    refresh()
    const interval = setInterval(refresh, 4000)
    return () => clearInterval(interval)
  }, [refresh])

  return (
    <div>
      <div className="page-head">
        <h2>Pending Enrollments</h2>
      </div>
      {error && <div className="form-error">{error}</div>}

      {!candidates.length ? (
        <div className="empty">No pending enrollments.</div>
      ) : (
        <div className="cand-grid">
          {candidates.map((c) => (
            <CandidateCard key={c.candidate_id} candidate={c} onAction={refresh} />
          ))}
        </div>
      )}
    </div>
  )
}

function CandidateCard({ candidate, onAction }: { candidate: Candidate; onAction: () => void }) {
  const cropUrl = useCandidateCropUrl(candidate.candidate_id, candidate.crop_urls[0])
  const [showPicker, setShowPicker] = useState(false)

  async function handleApprove() {
    const name = prompt('Display name for this new person:')
    if (!name) return
    await approveCandidate(candidate.candidate_id, name)
    invalidatePersonsCache()
    onAction()
  }

  async function handleMergeSuggested() {
    if (!candidate.suggested_person_id) return
    if (!confirm(`Merge this candidate into ${candidate.suggested_name}?`)) return
    await mergeCandidate(candidate.candidate_id, candidate.suggested_person_id)
    onAction()
  }

  async function handleReject() {
    if (!confirm('Reject this candidate?')) return
    await rejectCandidate(candidate.candidate_id)
    onAction()
  }

  async function handlePick(personId: string, name: string) {
    if (!confirm(`Merge this candidate into ${name}?`)) return
    await mergeCandidate(candidate.candidate_id, personId)
    setShowPicker(false)
    onAction()
  }

  return (
    <div className="cand-card">
      <img src={cropUrl} alt="face crop" />
      <div className="cand-body">
        <div className="meta">
          cam {candidate.camera_id} · track {candidate.track_id} · last seen{' '}
          {new Date(candidate.last_seen_at).toLocaleTimeString()}
        </div>
        {candidate.suggested_name ? (
          <div className="suggestion">
            Looks like {candidate.suggested_name} ({candidate.suggested_similarity})
          </div>
        ) : (
          <div className="suggestion" style={{ color: '#6b7280' }}>
            No gallery match
          </div>
        )}
        <div className="cand-actions">
          <button className="btn-approve" onClick={handleApprove}>
            New person
          </button>
          {candidate.suggested_person_id && (
            <button className="btn-merge" onClick={handleMergeSuggested}>
              Merge → {candidate.suggested_name}
            </button>
          )}
          <button className="btn-merge-other" onClick={() => setShowPicker(true)}>
            Merge to other…
          </button>
          <button className="btn-reject" onClick={handleReject}>
            Reject
          </button>
        </div>
      </div>
      {showPicker && <MergePickerModal onClose={() => setShowPicker(false)} onPick={handlePick} />}
    </div>
  )
}

function MergePickerModal({
  onClose,
  onPick,
}: {
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
          <strong>Merge into…</strong>
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
