import { useQuery } from '@tanstack/react-query'
import { AlertTriangle, CheckCircle, Activity } from 'lucide-react'
import { fetchPanels } from '../api/client'
import PanelCard from '../components/PanelCard'

export default function Dashboard() {
  const { data: panels, isLoading, error } = useQuery({
    queryKey: ['panels'],
    queryFn: fetchPanels,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-800">Failed to load panels. Please try again.</p>
      </div>
    )
  }

  // Mock stats for demo
  const stats = {
    totalPanels: panels?.length || 0,
    healthyPanels: Math.floor((panels?.length || 0) * 0.7),
    alertsToday: Math.floor(Math.random() * 5),
  }

  return (
    <div className="space-y-6">
      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-blue-100 rounded-lg">
              <Activity className="h-6 w-6 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Total Panels</p>
              <p className="text-2xl font-bold">{stats.totalPanels}</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-green-100 rounded-lg">
              <CheckCircle className="h-6 w-6 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Healthy Panels</p>
              <p className="text-2xl font-bold">{stats.healthyPanels}</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-red-100 rounded-lg">
              <AlertTriangle className="h-6 w-6 text-red-600" />
            </div>
            <div>
              <p className="text-sm text-gray-500">Alerts Today</p>
              <p className="text-2xl font-bold">{stats.alertsToday}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Panel Grid */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Monitored Panels
        </h2>
        {panels && panels.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {panels.map((panel) => (
              <PanelCard
                key={panel.id}
                panel={panel}
                latestRisk={['Low', 'Medium', 'High'][Math.floor(Math.random() * 3)] as any}
                latestSeverity={Math.random() * 0.8}
              />
            ))}
          </div>
        ) : (
          <div className="bg-white rounded-lg shadow p-8 text-center">
            <p className="text-gray-500">No panels registered yet.</p>
            <p className="text-sm text-gray-400 mt-2">
              Add panels to start monitoring their health.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}
