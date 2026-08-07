import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import './index.css'
import App from './App.tsx'
import Landing from './pages/Landing.tsx'
import Gallery from './pages/Gallery.tsx'
import GalleryRunDetail from './pages/GalleryRunDetail.tsx'
import NewRun from './pages/NewRun.tsx'
import SandboxRunResult from './pages/SandboxRunResult.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<App />}>
          <Route index element={<Landing />} />
          <Route path="gallery" element={<Gallery />} />
          <Route path="gallery/:directory" element={<GalleryRunDetail />} />
          <Route path="runs/new" element={<NewRun />} />
          <Route path="runs/:runId" element={<SandboxRunResult />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
