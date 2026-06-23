// Mirrors static/dashboard/index.html's authHeaders()/sessionStorage
// pattern, so logging in once on /dashboard carries over here with no
// second login screen.

export interface CameraLaunchArgs {
  fps: number
  det_size: number
  det_thresh: number
  reid_thresh: number
  min_face: number
  recognize: boolean
  persist_events: boolean
  cpu: boolean
}

export interface CameraStatus {
  camera_id: string
  display_name: string
  is_active: boolean
  status: string
  zone_id: string | null
  location_desc: string | null
  gpu_id: number | null
  desired_state: string
  process_state: string | null
  config_version: number
}

export interface CameraDetail extends CameraStatus {
  stream_url: string
  resolution_w: number
  resolution_h: number
  fps_target: number
  launch_args: CameraLaunchArgs
}

export interface NewCameraInput {
  camera_id: string
  display_name: string
  stream_url: string
  zone_id?: string | null
  location_desc?: string | null
  resolution_w?: number
  resolution_h?: number
  fps_target?: number
  gpu_id?: number | null
  launch_args?: Partial<CameraLaunchArgs>
}

export interface GpuStat {
  gpu_id: number
  name: string
  utilization_pct: number
  memory_used_mb: number
  memory_total_mb: number
  temperature_c: number | null
  power_w: number | null
}

export interface DiskStat {
  path: string
  used_gb: number
  total_gb: number
}

export const ZONE_TYPES = ['general', 'restricted', 'entrance', 'exit', 'monitored'] as const

export interface ZoneOccupant {
  person_id: string
  display_name: string
}

export interface ZoneStatus {
  zone_id: string
  display_name: string
  zone_type: string
  description: string | null
  occupant_count: number
  occupants: ZoneOccupant[]
  object_counts: Record<string, number>
}

export interface ZoneUpdate {
  display_name?: string
  zone_type?: string
  description?: string
}

export interface PresenceLogEntry {
  person_id: string | null
  display_name: string
  camera_id: string | null
  entry_time: string
  exit_time: string | null
  is_unknown: boolean
}

export interface Candidate {
  candidate_id: string
  track_id: string
  camera_id: string
  first_seen_at: string
  last_seen_at: string
  crop_count: number
  crop_urls: string[]
  status: string
  suggested_person_id: string | null
  suggested_name: string | null
  suggested_similarity: number | null
}

export interface Person {
  person_id: string
  display_name: string
  role: string | null
  enrolled_at: string
  is_active: boolean
  last_seen_at: string | null
  last_seen_zone: string | null
}

export interface SystemStats {
  cpu_pct: number
  memory_used_gb: number
  memory_total_gb: number
  disks: DiskStat[]
}

export function getToken(): string | null {
  return sessionStorage.getItem('te_token')
}

function authHeaders(): Record<string, string> {
  const token = getToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

// The access token is short-lived (15 min) by design; a single shared
// in-flight refresh avoids every concurrent 401'd request racing its own
// /refresh call. Falls back to the login bounce if there's no refresh
// token or it's also expired (7-day lifetime). Exported so the WebSocket
// layer can reuse it: a long-lived WS rejected at the handshake (close
// 4401) needs the same fresh token before it reconnects.
let refreshPromise: Promise<boolean> | null = null

export function refreshAccessToken(): Promise<boolean> {
  const refreshToken = sessionStorage.getItem('te_refresh')
  if (!refreshToken) return Promise.resolve(false)
  if (!refreshPromise) {
    refreshPromise = fetch('/api/v1/auth/refresh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken }),
    })
      .then(async (res) => {
        if (!res.ok) return false
        const data = await res.json()
        sessionStorage.setItem('te_token', data.access_token)
        sessionStorage.setItem('te_refresh', data.refresh_token)
        return true
      })
      .catch(() => false)
      .finally(() => {
        refreshPromise = null
      })
  }
  return refreshPromise
}

async function apiFetch<T>(path: string, options: RequestInit = {}, _retried = false): Promise<T> {
  const res = await fetch(path, {
    ...options,
    headers: {
      ...authHeaders(),
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...options.headers,
    },
  })
  if (res.status === 401) {
    if (!_retried && (await refreshAccessToken())) {
      return apiFetch<T>(path, options, true)
    }
    sessionStorage.removeItem('te_token')
    sessionStorage.removeItem('te_refresh')
    window.location.href = '/settings/login'
    throw new Error('Unauthorized')
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}) as { detail?: string })
    throw new Error(body.detail || res.statusText)
  }
  if (res.status === 204) return undefined as T
  return res.json() as Promise<T>
}

export const listCameras = () => apiFetch<CameraStatus[]>('/api/v1/cameras')
export const getCamera = (id: string) => apiFetch<CameraDetail>(`/api/v1/cameras/${id}`)
export const createCamera = (body: NewCameraInput) =>
  apiFetch<CameraDetail>('/api/v1/cameras', { method: 'POST', body: JSON.stringify(body) })
export const updateCamera = (id: string, body: Record<string, unknown>) =>
  apiFetch<CameraDetail>(`/api/v1/cameras/${id}`, { method: 'PATCH', body: JSON.stringify(body) })
export const startCamera = (id: string) => apiFetch(`/api/v1/cameras/${id}/start`, { method: 'POST' })
export const stopCamera = (id: string) => apiFetch(`/api/v1/cameras/${id}/stop`, { method: 'POST' })
export const deleteCamera = (id: string) => apiFetch(`/api/v1/cameras/${id}`, { method: 'DELETE' })
export const getCameraLog = (id: string) =>
  apiFetch<{ lines: string[]; message?: string }>(`/api/v1/cameras/${id}/log?lines=80`)

export const listZones = () => apiFetch<ZoneStatus[]>('/api/v1/zones')
export const createZone = (body: { zone_id: string; display_name: string; zone_type: string; description?: string }) =>
  apiFetch('/api/v1/zones', { method: 'POST', body: JSON.stringify(body) })
export const updateZone = (zoneId: string, body: ZoneUpdate) =>
  apiFetch(`/api/v1/zones/${zoneId}`, { method: 'PATCH', body: JSON.stringify(body) })
export const deleteZone = (zoneId: string) =>
  apiFetch(`/api/v1/zones/${zoneId}`, { method: 'DELETE' })
export const getPresenceLog = (zoneId: string) =>
  apiFetch<PresenceLogEntry[]>(`/api/v1/zones/${zoneId}/presence-log`)

export const listCandidates = () =>
  apiFetch<{ candidates: Candidate[] }>('/api/v1/identities/candidates?status=pending')
export const approveCandidate = (candidateId: string, displayName: string) =>
  apiFetch(`/api/v1/identities/candidates/${candidateId}/approve`, {
    method: 'POST',
    body: JSON.stringify({ display_name: displayName }),
  })
export const mergeCandidate = (candidateId: string, personId: string) =>
  apiFetch(`/api/v1/identities/candidates/${candidateId}/approve`, {
    method: 'POST',
    body: JSON.stringify({ person_id: personId }),
  })
export const rejectCandidate = (candidateId: string) =>
  apiFetch(`/api/v1/identities/candidates/${candidateId}/reject`, { method: 'POST' })
export const listIdentities = () => apiFetch<Person[]>('/api/v1/identities')

// <img src> can't carry an Authorization header, and unlike camera frames
// this isn't a continuous stream worth a dedicated endpoint - fetch once
// as an authenticated blob and hand back an object URL. Caller owns
// caching (see CandidatesPage's module-level cropUrlCache).
export async function fetchCropBlobUrl(cropUrl: string): Promise<string | null> {
  const res = await fetch(cropUrl, { headers: { Authorization: `Bearer ${getToken() || ''}` } })
  if (!res.ok) return null
  const blob = await res.blob()
  return URL.createObjectURL(blob)
}

export const listGpus = () => apiFetch<GpuStat[]>('/api/v1/hardware/gpus')
export const getSystemStats = () => apiFetch<SystemStats>('/api/v1/hardware/system')

export function wsUrl(path: string): string {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const token = getToken() || ''
  return `${proto}://${location.host}${path}?token=${encodeURIComponent(token)}`
}

// multipart/x-mixed-replace - browsers render this natively in an <img>,
// no JS polling. Call once per camera_id and never recompute the src from
// state that changes on a poll tick, or the stream restarts every time
// (see CameraTile's comment).
export function mjpegUrl(cameraId: string): string {
  const token = getToken() || ''
  return `/api/v1/cameras/${cameraId}/mjpeg?token=${encodeURIComponent(token)}`
}

export interface EnrollPoseBucket {
  key: string
  label: string
  needed: number
}

export interface EnrollSession {
  session_id: string
  pose_buckets: EnrollPoseBucket[]
}

export interface EnrollFrameResult {
  face_found: boolean
  yaw: number | null
  pitch: number | null
  bucket: string | null
  captured: boolean
  progress: number
  total: number
  pending: string[]
  done: boolean
}

export const createEnrollSession = (crops = 10) =>
  apiFetch<EnrollSession>(`/api/v1/manual-enroll/sessions?crops=${crops}`, { method: 'POST' })
export const submitEnrollFrame = (sessionId: string, imageB64: string) =>
  apiFetch<EnrollFrameResult>(`/api/v1/manual-enroll/sessions/${sessionId}/frame`, {
    method: 'POST',
    body: JSON.stringify({ image_b64: imageB64 }),
  })
export const discardEnrollSession = (sessionId: string) =>
  apiFetch(`/api/v1/manual-enroll/sessions/${sessionId}`, { method: 'DELETE' })
export const finalizeEnrollSession = (
  sessionId: string,
  body: { display_name?: string; role?: string; person_id?: string },
) =>
  apiFetch<Person>(`/api/v1/manual-enroll/sessions/${sessionId}/finalize`, {
    method: 'POST',
    body: JSON.stringify(body),
  })
