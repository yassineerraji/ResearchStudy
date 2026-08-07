// Route wrapper: one curated Results Gallery experiment, identified by its
// `outputs/` directory name. All the actual rendering lives in
// `ExperimentDetailView`; this file only wires it to the gallery API.

import { useParams } from 'react-router-dom'
import { getExperimentDetail, getReplaySlice } from '../api/client'
import ExperimentDetailView from '../components/ExperimentDetailView'

export default function GalleryRunDetail() {
  const { directory } = useParams<{ directory: string }>()
  if (!directory) return null

  return (
    <ExperimentDetailView
      key={directory}
      fetchDetail={() => getExperimentDetail(directory)}
      fetchReplay={(replication, policy, runKind) =>
        getReplaySlice(directory, replication, policy, runKind)
      }
      backTo="/gallery"
      backLabel="← Results Gallery"
    />
  )
}
