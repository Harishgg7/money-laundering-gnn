import React from 'react';

interface MetricCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  highlight?: boolean;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subtext,
  highlight = false,
}) => {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div
        className="metric-value tabular"
        style={highlight ? { color: 'var(--risk-high-text)' } : undefined}
      >
        {value}
      </div>
      {subtext && <div className="metric-sub">{subtext}</div>}
    </div>
  );
};
