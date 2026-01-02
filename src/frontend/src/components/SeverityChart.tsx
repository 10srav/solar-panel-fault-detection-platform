import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import type { FaultEvent } from '../types'
import { format } from 'date-fns'

interface SeverityChartProps {
  events: FaultEvent[]
}

export default function SeverityChart({ events }: SeverityChartProps) {
  const data = events
    .slice()
    .reverse()
    .map((event) => ({
      date: format(new Date(event.detected_at), 'MM/dd HH:mm'),
      severity: event.severity_score,
      faultClass: event.fault_class,
    }))

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        Severity History
      </h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" fontSize={12} />
            <YAxis domain={[0, 1]} fontSize={12} />
            <Tooltip
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload
                  return (
                    <div className="bg-white shadow-lg rounded p-3 border">
                      <p className="text-sm text-gray-500">{data.date}</p>
                      <p className="font-semibold">
                        Severity: {(data.severity * 100).toFixed(1)}%
                      </p>
                      <p className="text-sm text-gray-600">
                        Class: {data.faultClass}
                      </p>
                    </div>
                  )
                }
                return null
              }}
            />
            <ReferenceLine
              y={0.3}
              stroke="#10b981"
              strokeDasharray="3 3"
              label={{ value: 'Low', position: 'right', fontSize: 10 }}
            />
            <ReferenceLine
              y={0.7}
              stroke="#ef4444"
              strokeDasharray="3 3"
              label={{ value: 'High', position: 'right', fontSize: 10 }}
            />
            <Line
              type="monotone"
              dataKey="severity"
              stroke="#3b82f6"
              strokeWidth={2}
              dot={{ fill: '#3b82f6' }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
