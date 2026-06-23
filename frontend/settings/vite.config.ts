import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Served at /settings/ in production (FastAPI StaticFiles mount over the
// built dist/ - see src/api/main.py), so asset URLs need that base path.
// Dev server proxies /api straight to uvicorn so the app can call the same
// relative paths in both dev and prod with no env-var branch.
export default defineConfig({
  plugins: [react()],
  base: '/settings/',
  server: {
    proxy: {
      '/api': { target: 'http://localhost:8000', ws: true },
    },
  },
})
