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

async function apiFetch<T>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(path, {
    ...options,
    headers: {
      ...authHeaders(),
      ...(options.body ? { 'Content-Type': 'application/json' } : {}),
      ...options.headers,
    },
  })
  if (res.status === 401) {
    // No second login UI here - the dashboard's is already battle-tested.
    window.location.href = '/dashboard/'
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

export const listGpus = () => apiFetch<GpuStat[]>('/api/v1/hardware/gpus')
export const getSystemStats = () => apiFetch<SystemStats>('/api/v1/hardware/system')

export function wsUrl(path: string): string {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const token = getToken() || ''
  return `${proto}://${location.host}${path}?token=${encodeURIComponent(token)}`
}
