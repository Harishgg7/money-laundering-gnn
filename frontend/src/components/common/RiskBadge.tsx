import React from 'react';
import { RiskLevel } from '../../types';

interface RiskBadgeProps {
  level: RiskLevel;
  score?: number;
  showScore?: boolean;
  scoreFormat?: 'percent' | 'decimal' | 'both';
}

export const RiskBadge: React.FC<RiskBadgeProps> = ({
  level,
  score,
  showScore = false,
  scoreFormat = 'percent',
}) => {
  const levelClass =
    level === 'HIGH'
      ? 'badge-high'
      : level === 'MEDIUM'
      ? 'badge-med'
      : 'badge-low';

  const label =
    level === 'HIGH' ? 'HIGH RISK' : level === 'MEDIUM' ? 'MED RISK' : 'LOW RISK';

  let formattedScore = '';
  if (score !== undefined) {
    if (scoreFormat === 'decimal') {
      formattedScore = score.toFixed(2);
    } else if (scoreFormat === 'both') {
      formattedScore = `${Math.round(score * 100)}% • ${score.toFixed(2)}`;
    } else {
      formattedScore = `${Math.round(score * 100)}%`;
    }
  }

  return (
    <span className={`badge ${levelClass}`}>
      <span className="badge-dot" />
      <span>{label}</span>
      {showScore && score !== undefined && (
        <span className="mono tabular" style={{ opacity: 0.88, marginLeft: 3 }}>
          ({formattedScore})
        </span>
      )}
    </span>
  );
};
