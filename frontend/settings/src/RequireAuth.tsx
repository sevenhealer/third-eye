import { Navigate, Outlet } from 'react-router-dom'
import { getToken } from './api/client'

// One auth check, one level above the router's pages - not duplicated
// per-page. Client-side route transitions don't remount this, so it only
// re-evaluates on a real page load, same as the old single-effect check.
export function RequireAuth() {
  if (getToken() === null) {
    return <Navigate to="/login" replace />
  }
  return <Outlet />
}
