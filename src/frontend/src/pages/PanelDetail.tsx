import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { MapPin, Calendar, Zap, Clock } from 'lucide-react'
import { format } from 'date-fns'
import { fetchPanelHistory } from '../api/client'
import RiskBadge from '../components/RiskBadge'
import SeverityChart from '../components/SeverityChart'

export default function PanelDetail() {
  const { panelId } = useParams<{ panelId: string }>()

  const { data, isLoading, error } = useQuery({
    queryKey: ['panel-history', panelId],
    queryFn: () => fetchPanelHistory(panelId!),
    enabled: !!panelId,
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600" />
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-800">Failed to load panel details.</p>
      </div>
    )
  }

  const { panel, fault_events, total_events, high_risk_events, latest_severity } = data

  return (
    <div className="space-y-6">
      {/* Panel Info */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">{panel.name}</h2>
            {panel.location && (
              <p className="flex items-center gap-1 text-gray-500 mt-1">
                <MapPin className="h-4 w-4" />
                {panel.location}
              </p>
            )}
          </div>
          {latest_severity !== null && (
            <div className="text-right">
              <p className="text-sm text-gray-500">Current Severity</p>
              <p className="text-3xl font-bold">
                {(latest_severity * 100).toFixed(1)}%
              </p>
            </div>
          )}
        </div>

        <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-4">
          {panel.capacity_kw && (
            <div className="flex items-center gap-2 text-gray-600">
              <Zap className="h-5 w-5 text-yellow-500" />
              <div>
                <p className="text-sm text-gray-400">Capacity</p>
                <p className="font-semibold">{panel.capacity_kw} kW</p>
              </div>
            </div>
          )}
          {panel.installation_date && (
            <div className="flex items-center gap-2 text-gray-600">
              <Calendar className="h-5 w-5" />
              <div>
                <p className="text-sm text-gray-400">Installed</p>
                <p className="font-semibold">
                  {format(new Date(panel.installation_date), 'MMM d, yyyy')}
                </p>
              </div>
            </div>
          )}
          <div className="flex items-center gap-2 text-gray-600">
            <Clock className="h-5 w-5" />
            <div>
              <p className="text-sm text-gray-400">Total Events</p>
              <p className="font-semibold">{total_events}</p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-gray-600">
            <div className="h-5 w-5 bg-red-100 rounded-full flex items-center justify-center">
              <span className="text-xs text-red-600 font-bold">!</span>
            </div>
            <div>
              <p className="text-sm text-gray-400">High Risk Events</p>
              <p className="font-semibold">{high_risk_events}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Severity Chart */}
      {fault_events.length > 0 && <SeverityChart events={fault_events} />}

      {/* Event History */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b">
          <h3 className="text-lg font-semibold text-gray-900">Fault History</h3>
        </div>
        <div className="divide-y">
          {fault_events.length > 0 ? (
            fault_events.map((event) => (
              <div key={event.id} className="p-6 hover:bg-gray-50">
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-3">
                      <span className="font-semibold text-gray-900">
                        {event.fault_class}
                      </span>
                      <RiskBadge level={event.risk_level} size="sm" />
                    </div>
                    <p className="text-sm text-gray-500 mt-1">
                      Confidence: {(event.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-400">
                      {format(new Date(event.detected_at), 'MMM d, yyyy HH:mm')}
                    </p>
                    <p className="font-semibold mt-1">
                      Severity: {(event.severity_score * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="p-8 text-center text-gray-500">
              No fault events recorded.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
