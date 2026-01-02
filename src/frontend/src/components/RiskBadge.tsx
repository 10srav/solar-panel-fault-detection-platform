import clsx from 'clsx'

interface RiskBadgeProps {
  level: 'Low' | 'Medium' | 'High'
  size?: 'sm' | 'md' | 'lg'
}

export default function RiskBadge({ level, size = 'md' }: RiskBadgeProps) {
  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-sm',
    lg: 'px-3 py-1.5 text-base',
  }

  const levelClasses = {
    Low: 'bg-green-100 text-green-800',
    Medium: 'bg-yellow-100 text-yellow-800',
    High: 'bg-red-100 text-red-800',
  }

  return (
    <span
      className={clsx(
        'inline-flex items-center rounded-full font-medium',
        sizeClasses[size],
        levelClasses[level]
      )}
    >
      {level}
    </span>
  )
}
