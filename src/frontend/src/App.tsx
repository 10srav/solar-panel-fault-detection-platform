import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import PanelDetail from './pages/PanelDetail'
import Inference from './pages/Inference'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/panels/:panelId" element={<PanelDetail />} />
        <Route path="/inference" element={<Inference />} />
      </Routes>
    </Layout>
  )
}

export default App
