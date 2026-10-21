import React from 'react';
import { CategoryDistribution } from '../../types';

interface DistributionChartProps {
  distribution: CategoryDistribution[];
}

export const DistributionChart: React.FC<DistributionChartProps> = ({ distribution }) => {
  const size = 180;
  const strokeWidth = 24;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;

  // Clean, restrained palette matching institutional financial software
  const colors = ['#2563eb', '#64748b', '#94a3b8'];

  let accumulatedPercent = 0;

  return (
    <div className="panel-card" style={{ display: 'flex', flexDirection: 'column' }}>
      <div className="panel-title">
        <div>
          <span>Transaction Distribution</span>
          <div style={{ fontSize: 11, fontWeight: 400, color: 'var(--text-muted)', marginTop: 2 }}>
            Breakdown across transaction categories
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 24, padding: '10px 0' }}>
        {/* SVG Donut */}
        <div style={{ position: 'relative', width: size, height: size, flexShrink: 0 }}>
          <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ transform: 'rotate(-90deg)' }}>
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="transparent"
              stroke="var(--bg-surface-subtle)"
              strokeWidth={strokeWidth}
            />
            {distribution.map((item, index) => {
              const strokeDasharray = `${(item.percentage / 100) * circumference} ${circumference}`;
              const strokeDashoffset = -((accumulatedPercent / 100) * circumference);
              accumulatedPercent += item.percentage;

              return (
                <circle
                  key={item.category}
                  cx={size / 2}
                  cy={size / 2}
                  r={radius}
                  fill="transparent"
                  stroke={colors[index % colors.length]}
                  strokeWidth={strokeWidth}
                  strokeDasharray={strokeDasharray}
                  strokeDashoffset={strokeDashoffset}
                  strokeLinecap="butt"
                />
              );
            })}
          </svg>
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              pointerEvents: 'none',
            }}
          >
            <span style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Primary</span>
            <span style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)' }} className="tabular">
              {distribution[0] != null ? `${distribution[0].percentage}%` : '—'}
            </span>
          </div>
        </div>

        {/* Legend */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10, minWidth: 140 }}>
          {distribution.map((item, index) => (
            <div key={item.category} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span
                  style={{
                    width: 10,
                    height: 10,
                    borderRadius: 'var(--radius-xs)',
                    backgroundColor: colors[index % colors.length],
                    flexShrink: 0,
                  }}
                />
                <span style={{ fontSize: 12, color: 'var(--text-primary)', fontWeight: 500 }}>
                  {item.category}
                </span>
              </div>
              <span className="mono tabular" style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)' }}>
                {item.percentage}%
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
