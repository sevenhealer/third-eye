import { useCallback, useEffect, useRef, useState } from 'react'
import {
  type EnrollSession,
  type EnrollFrameResult,
  type Person,
  createEnrollSession,
  submitEnrollFrame,
  discardEnrollSession,
  finalizeEnrollSession,
} from '../api/client'
import { PersonPickerModal } from '../components/PersonPicker'
import { invalidatePersonsCache } from '../api/personsCache'

const FRAME_INTERVAL_MS = 350

type Phase = 'idle' | 'capturing' | 'review' | 'done'

export function EnrollPage() {
  const [phase, setPhase] = useState<Phase>('idle')
  const [session, setSession] = useState<EnrollSession | null>(null)
  const [frameResult, setFrameResult] = useState<EnrollFrameResult | null>(null)
  const [error, setError] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [role, setRole] = useState('visitor')
  const [showPicker, setShowPicker] = useState(false)
  const [enrolledPerson, setEnrolledPerson] = useState<Person | null>(null)
  const [submitting, setSubmitting] = useState(false)

  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const sessionIdRef = useRef<string | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopCamera = useCallback(() => {
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
  }, [])

  // Leaving mid-capture shouldn't strand server-side session state.
  useEffect(() => {
    return () => {
      stopCamera()
      if (sessionIdRef.current) discardEnrollSession(sessionIdRef.current).catch(() => {})
    }
  }, [stopCamera])

  async function handleStart() {
    setError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
    } catch {
      setError('Could not access the camera — check browser permissions.')
      return
    }

    try {
      const s = await createEnrollSession()
      setSession(s)
      sessionIdRef.current = s.session_id
      setFrameResult(null)
      setPhase('capturing')
      intervalRef.current = setInterval(captureFrame, FRAME_INTERVAL_MS)
    } catch (e) {
      stopCamera()
      setError(e instanceof Error ? e.message : 'Failed to start enrollment session.')
    }
  }

  async function captureFrame() {
    const video = videoRef.current
    const canvas = canvasRef.current
    const sessionId = sessionIdRef.current
    if (!video || !canvas || !sessionId || video.videoWidth === 0) return

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
    const dataUrl = canvas.toDataURL('image/jpeg', 0.85)

    try {
      const result = await submitEnrollFrame(sessionId, dataUrl)
      setFrameResult(result)
      if (result.done) {
        if (intervalRef.current !== null) {
          clearInterval(intervalRef.current)
          intervalRef.current = null
        }
        stopCamera()
        setPhase('review')
      }
    } catch {
      // a single dropped frame isn't worth surfacing - the next tick retries
    }
  }

  async function handleCancel() {
    stopCamera()
    if (sessionIdRef.current) {
      await discardEnrollSession(sessionIdRef.current).catch(() => {})
      sessionIdRef.current = null
    }
    setSession(null)
    setFrameResult(null)
    setPhase('idle')
  }

  async function finalize(body: { display_name?: string; role?: string; person_id?: string }) {
    if (!sessionIdRef.current) return
    setSubmitting(true)
    setError('')
    try {
      const person = await finalizeEnrollSession(sessionIdRef.current, body)
      invalidatePersonsCache()
      sessionIdRef.current = null
      setEnrolledPerson(person)
      setPhase('done')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Enrollment failed.')
    } finally {
      setSubmitting(false)
    }
  }

  function handleEnrollNew(e: React.FormEvent) {
    e.preventDefault()
    if (!displayName.trim()) return
    finalize({ display_name: displayName.trim(), role })
  }

  function handlePickExisting(personId: string) {
    setShowPicker(false)
    finalize({ person_id: personId })
  }

  function reset() {
    setSession(null)
    setFrameResult(null)
    setDisplayName('')
    setRole('visitor')
    setEnrolledPerson(null)
    setError('')
    setPhase('idle')
  }

  const labelByKey = Object.fromEntries((session?.pose_buckets ?? []).map((b) => [b.key, b.label]))
  const pending = frameResult?.pending ?? session?.pose_buckets.map((b) => b.key) ?? []
  const currentTarget = pending[0]
  const progress = frameResult?.progress ?? 0
  const total = frameResult?.total ?? session?.pose_buckets.reduce((n, b) => n + b.needed, 0) ?? 0

  return (
    <div>
      <div className="page-head">
        <h2>Enroll Person</h2>
      </div>
      {error && <div className="form-error">{error}</div>}

      {phase === 'idle' && (
        <div className="panel">
          <p className="hint" style={{ marginTop: 0 }}>
            Guided capture: hold the device so the person's face is visible, then slowly turn
            through each prompted pose. Produces a higher-quality embedding than a single
            passing camera sighting — use this for deliberately enrolling someone (e.g. a new
            family member) rather than waiting for them to show up as a Pending Enrollment.
          </p>
          <button className="btn-primary" onClick={handleStart}>
            Start capture
          </button>
        </div>
      )}

      {(phase === 'capturing' || phase === 'review') && (
        <div className="panel enroll-capture">
          <video ref={videoRef} className="enroll-video" muted playsInline />
          <canvas ref={canvasRef} style={{ display: 'none' }} />

          {phase === 'capturing' && (
            <div className="enroll-status">
              <div className="enroll-progress-bar">
                <div
                  className="enroll-progress-fill"
                  style={{ width: total ? `${Math.min(100, (progress / total) * 100)}%` : '0%' }}
                />
              </div>
              <div className="enroll-progress-label">
                {progress}/{total} crops captured
              </div>
              {currentTarget ? (
                <div className="enroll-prompt">{labelByKey[currentTarget] ?? currentTarget}</div>
              ) : (
                <div className="enroll-prompt">Almost done…</div>
              )}
              <div className="hint">
                {frameResult?.face_found === false ? 'No face detected' : ' '}
              </div>
              <div className="enroll-pose-list">
                {(session?.pose_buckets ?? []).map((b) => (
                  <span
                    key={b.key}
                    className={`badge${pending.includes(b.key) ? '' : ' badge-done'}`}
                  >
                    {pending.includes(b.key) ? '' : '✓ '}
                    {b.label.replace(/^Look |^Turn your head |^Tilt your head /, '')}
                  </span>
                ))}
              </div>
              <button className="btn-danger" onClick={handleCancel} style={{ marginTop: 10 }}>
                Cancel
              </button>
            </div>
          )}

          {phase === 'review' && (
            <div className="enroll-status">
              <div className="enroll-prompt">Capture complete — {progress} crops</div>
              <form onSubmit={handleEnrollNew} className="enroll-finalize-form">
                <input
                  placeholder="Display name"
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                />
                <input
                  placeholder="Role (e.g. visitor, family, staff)"
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                />
                <button className="btn-primary" type="submit" disabled={submitting || !displayName.trim()}>
                  Enroll as new person
                </button>
              </form>
              <button
                className="btn-merge-other"
                style={{ marginTop: 8 }}
                disabled={submitting}
                onClick={() => setShowPicker(true)}
              >
                Add to an existing person…
              </button>
              <button className="btn-danger" style={{ marginTop: 8 }} onClick={handleCancel} disabled={submitting}>
                Discard capture
              </button>
            </div>
          )}
        </div>
      )}

      {phase === 'done' && enrolledPerson && (
        <div className="panel">
          <p>
            Enrolled <strong>{enrolledPerson.display_name}</strong> ({enrolledPerson.role ?? 'visitor'}).
          </p>
          <button className="btn-primary" onClick={reset}>
            Enroll another
          </button>
        </div>
      )}

      {showPicker && (
        <PersonPickerModal
          title="Add this capture to…"
          onClose={() => setShowPicker(false)}
          onPick={(personId) => handlePickExisting(personId)}
        />
      )}
    </div>
  )
}
