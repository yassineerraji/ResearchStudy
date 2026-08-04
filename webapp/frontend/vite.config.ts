import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxies /api during `npm run dev` to the FastAPI backend (M0's
// uvicorn app.main:app, default port 8000) so the frontend can call
// same-origin relative paths (`/api/v1/...`) in both dev and production,
// instead of hardcoding a backend origin that changes per deployment.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
