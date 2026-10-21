import React from 'react';
import { AMLPattern } from '../../types';

interface PatternCardProps {
  patterns: AMLPattern[];
}

export const PatternCard: React.FC<PatternCardProps> = ({ patterns }) => {
  return (
    <div className="panel-card" style={{ marginBottom: 16 }}>
      <div className="panel-title">
        <span>Observed Transaction Patterns</span>
        <span style={{ fontSize: 11, fontWeight: 400, color: 'var(--text-muted)' }}>
          Topological pattern heuristics
        </span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {patterns.map((p) => {
          const isObserved = p.status === 'Observed' || p.status === 'Detected';
          const isPossible = p.status === 'Possible';
          const displayStatus = p.status === 'Detected' ? 'Observed' : p.status === 'Not detected' ? 'Not observed' : p.status;

          return (
            <div
              key={p.name}
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                justifyContent: 'space-between',
                padding: '8px 10px',
                backgroundColor: isObserved ? 'var(--risk-high-bg)' : isPossible ? 'var(--risk-med-bg)' : 'var(--bg-surface-subtle)',
                border: `1px solid ${isObserved ? 'var(--risk-high-border)' : isPossible ? 'var(--risk-med-border)' : 'var(--border-subtle)'}`,
                borderRadius: 'var(--radius-xs)',
              }}
            >
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontWeight: 600, fontSize: 12, color: 'var(--text-primary)' }}>
                    {p.name}
                  </span>
                  <span
                    style={{
                      fontSize: 10,
                      fontWeight: 600,
                      padding: '1px 6px',
                      borderRadius: 10,
                      backgroundColor: isObserved ? 'var(--risk-high-dot)' : isPossible ? 'var(--risk-med-dot)' : '#94a3b8',
                      color: '#ffffff',
                      letterSpacing: '0.02em',
                    }}
                  >
                    {displayStatus}
                  </span>
                </div>
                <div style={{ fontSize: 11.5, color: 'var(--text-secondary)', marginTop: 3 }}>
                  {p.detail}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
