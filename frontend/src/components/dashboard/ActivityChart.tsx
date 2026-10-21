import React, { useState } from 'react';
import { ActivityDataPoint, TimeRangeFilter } from '../../types';

interface ActivityChartProps {
  data: ActivityDataPoint[];
  timeRange: TimeRangeFilter;
  onTimeRangeChange: (range: TimeRangeFilter) => void;
}

export const ActivityChart: React.FC<ActivityChartProps> = ({
  data,
  timeRange,
  onTimeRangeChange,
}) => {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);

  const ranges: TimeRangeFilter[] = ['7D', '30D', '3M', '1Y', 'CUSTOM'];
  const labels: Record<TimeRangeFilter, string> = {
    '7D': '7 Days',
    '30D': '30 Days',
    '3M': '3 Months',
    '1Y': '1 Year',
    'CUSTOM': 'Custom',
  };

  const width = 640;
  const height = 210;
  const padding = { top: 20, right: 20, bottom: 30, left: 55 };

  const chartW = width - padding.left - padding.right;
  const chartH = height - padding.top - padding.bottom;

  const maxVal = Math.max(...data.map((d) => d.volume), 1000000) * 1.15;

  const points = data.map((d, i) => {
    const x = padding.left + (i / Math.max(data.length - 1, 1)) * chartW;
    const y = padding.top + chartH - (d.volume / maxVal) * chartH;
    const flaggedY = padding.top + chartH - (d.flaggedVolume / maxVal) * chartH;
    return { x, y, flaggedY, d };
  });

  const pathD = points.reduce((acc, p, i) => {
    return i === 0 ? `M ${p.x} ${p.y}` : `${acc} L ${p.x} ${p.y}`;
  }, '');

  const areaD = `${pathD} L ${points[points.length - 1]?.x || 0} ${padding.top + chartH} L ${points[0]?.x || 0} ${padding.top + chartH} Z`;

  const flaggedPathD = points.reduce((acc, p, i) => {
    return i === 0 ? `M ${p.x} ${p.flaggedY}` : `${acc} L ${p.x} ${p.flaggedY}`;
  }, '');

  const formatCurrency = (val: number) => {
    if (val >= 10000000) return `₹${(val / 10000000).toFixed(1)}Cr`;
    if (val >= 100000) return `₹${(val / 100000).toFixed(1)}L`;
    if (val >= 1000000) return `$${(val / 1000000).toFixed(1)}M`;
    if (val >= 1000) return `$${(val / 1000).toFixed(0)}k`;
    return `$${val}`;
  };

  return (
    <div className="panel-card" style={{ display: 'flex', flexDirection: 'column' }}>
      <div className="panel-title">
        <div>
          <span>Transaction Activity Over Time</span>
          <div style={{ fontSize: 11, fontWeight: 400, color: 'var(--text-muted)', marginTop: 2 }}>
            Aggregated volume and flagged activity across monitored accounts
          </div>
        </div>
        {/* Time Filter Controls */}
        <div style={{ display: 'flex', gap: 4 }}>
          {ranges.map((r) => (
            <button
              key={r}
              className={`btn btn-sm ${timeRange === r ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => onTimeRangeChange(r)}
              style={{ padding: '2px 8px', fontSize: 11 }}
            >
              {labels[r]}
            </button>
          ))}
        </div>
      </div>

      <div style={{ position: 'relative', width: '100%', height: height }}>
        <svg
          viewBox={`0 0 ${width} ${height}`}
          style={{ width: '100%', height: '100%', overflow: 'visible' }}
          onMouseLeave={() => setHoverIndex(null)}
        >
          {/* Horizontal Gridlines */}
          {[0, 0.33, 0.66, 1].map((ratio) => {
            const y = padding.top + chartH * (1 - ratio);
            const val = maxVal * ratio;
            return (
              <g key={ratio}>
                <line
                  x1={padding.left}
                  y1={y}
                  x2={width - padding.right}
                  y2={y}
                  stroke="var(--border-subtle)"
                  strokeDasharray={ratio === 0 ? undefined : '3,3'}
                />
                <text
                  x={padding.left - 8}
                  y={y + 3.5}
                  textAnchor="end"
                  fontSize="10"
                  fill="var(--text-muted)"
                  className="mono tabular"
                >
                  {formatCurrency(val)}
                </text>
              </g>
            );
          })}

          {/* Area Fill */}
          <path d={areaD} fill="var(--primary-blue-subtle)" opacity="0.6" />

          {/* Total Volume Line */}
          <path
            d={pathD}
            fill="none"
            stroke="var(--primary-blue)"
            strokeWidth="2"
            strokeLinejoin="round"
          />

          {/* Flagged Volume Line */}
          <path
            d={flaggedPathD}
            fill="none"
            stroke="var(--risk-high-text)"
            strokeWidth="1.5"
            strokeDasharray="4,3"
            strokeLinejoin="round"
          />

          {/* Points & Interactive Tooltips */}
          {points.map((p, i) => (
            <g
              key={i}
              onMouseEnter={() => setHoverIndex(i)}
              style={{ cursor: 'pointer' }}
            >
              {/* Invisible wide hover target */}
              <rect
                x={p.x - 15}
                y={padding.top}
                width={30}
                height={chartH}
                fill="transparent"
              />
              <circle
                cx={p.x}
                cy={p.y}
                r={hoverIndex === i ? 4.5 : 2.5}
                fill="#ffffff"
                stroke="var(--primary-blue)"
                strokeWidth="2"
              />
              <text
                x={p.x}
                y={padding.top + chartH + 18}
                textAnchor="middle"
                fontSize="10.5"
                fill={hoverIndex === i ? 'var(--text-primary)' : 'var(--text-muted)'}
                fontWeight={hoverIndex === i ? '600' : '400'}
              >
                {p.d.label}
              </text>
            </g>
          ))}
        </svg>

        {/* Hover Tooltip Box */}
        {hoverIndex !== null && points[hoverIndex] && (
          <div
            style={{
              position: 'absolute',
              left: `${(points[hoverIndex].x / width) * 100}%`,
              top: `${Math.max(10, (points[hoverIndex].y / height) * 100 - 35)}%`,
              transform: 'translate(-50%, -100%)',
              background: '#ffffff',
              border: '1px solid var(--border-default)',
              borderRadius: 'var(--radius-xs)',
              padding: '6px 10px',
              boxShadow: 'var(--shadow-sm)',
              pointerEvents: 'none',
              zIndex: 10,
              fontSize: 11,
              whiteSpace: 'nowrap',
            }}
          >
            <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: 2 }}>
              {points[hoverIndex].d.label}
            </div>
            <div style={{ color: 'var(--primary-blue)' }} className="tabular">
              Total: {formatCurrency(points[hoverIndex].d.volume)} ({points[hoverIndex].d.transactionCount.toLocaleString()} txns)
            </div>
            <div style={{ color: 'var(--risk-high-text)' }} className="tabular">
              Flagged: {formatCurrency(points[hoverIndex].d.flaggedVolume)}
            </div>
          </div>
        )}
      </div>

      {/* Chart Legend */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 16,
          marginTop: 10,
          paddingTop: 8,
          borderTop: '1px solid var(--border-subtle)',
          fontSize: 11,
          color: 'var(--text-secondary)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 12, height: 2.5, backgroundColor: 'var(--primary-blue)' }} />
          <span>Total Transaction Volume</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 12, height: 2, borderTop: '2px dashed var(--risk-high-text)' }} />
          <span>Flagged Suspicious Activity</span>
        </div>
      </div>
    </div>
  );
};
