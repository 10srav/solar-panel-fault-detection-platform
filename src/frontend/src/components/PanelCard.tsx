import { Link } from 'react-router-dom'
import { MapPin, Calendar, Zap, AlertTriangle } from 'lucide-react'
import type { Panel } from '../types'
import RiskBadge from './RiskBadge'

interface PanelCardProps {
  panel: Panel
  latestRisk?: 'Low' | 'Medium' | 'High'
  latestSeverity?: number
}

export default function PanelCard({ panel, latestRisk, latestSeverity }: PanelCardProps) {
  return (
    <Link
      to={`/panels/${panel.id}`}
      className="block bg-white rounded-lg shadow hover:shadow-md transition-shadow p-6"
    >
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">{panel.name}</h3>
          {panel.location && (
            <p className="flex items-center gap-1 text-sm text-gray-500 mt-1">
              <MapPin className="h-4 w-4" />
              {panel.location}
            </p>
          )}
        </div>
        {latestRisk && <RiskBadge level={latestRisk} />}
      </div>

      <div className="mt-4 grid grid-cols-2 gap-4">
        {panel.capacity_kw && (
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <Zap className="h-4 w-4 text-yellow-500" />
            <span>{panel.capacity_kw} kW</span>
          </div>
        )}
        {panel.installation_date && (
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <Calendar className="h-4 w-4" />
            <span>{new Date(panel.installation_date).toLocaleDateString()}</span>
          </div>
        )}
      </div>

      {latestSeverity !== undefined && (
        <div className="mt-4 pt-4 border-t">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-500">Severity Score</span>
            <span className="font-semibold">{(latestSeverity * 100).toFixed(1)}%</span>
          </div>
          <div className="mt-2 h-2 bg-gray-200 rounded-full overflow-hidden">
            <div
              className={clsx(
                'h-full rounded-full transition-all',
                latestSeverity < 0.3
                  ? 'bg-green-500'
                  : latestSeverity < 0.7
                  ? 'bg-yellow-500'
                  : 'bg-red-500'
              )}
              style={{ width: `${latestSeverity * 100}%` }}
            />
          </div>
        </div>
      )}
    </Link>
  )
}

function clsx(...classes: (string | boolean | undefined)[]) {
  return classes.filter(Boolean).join(' ')
}
