import React from 'react';
import { useNavigate } from 'react-router-dom';
import { NetworkNode } from '../../types';
import { RiskBadge } from '../common/RiskBadge';
import { ArrowUpRight, X, Focus } from 'lucide-react';

interface AccountPanelProps {
  node: NetworkNode | null;
  onClose: () => void;
  onCenterAccount?: (accountId: string) => void;
}

export const AccountPanel: React.FC<AccountPanelProps> = ({ node, onClose, onCenterAccount }) => {
  const navigate = useNavigate();

  if (!node) return null;

  const formatCurrency = (val: number) => {
    if (val >= 10000000) return `₹${(val / 10000000).toFixed(2)} Cr`;
    if (val >= 100000) return `₹${(val / 100000).toFixed(2)} L`;
    return `$${val.toLocaleString()}`;
  };

  const handleInvestigate = () => {
    // Navigate dynamically to the account's investigation dossier (no hardcoded case ID mapping)
    navigate(`/investigations/${encodeURIComponent(node.accountId)}`);
  };

  const relationshipLabel = node.isTarget
    ? 'Subject Account'
    : (node as any).hopLevel === 1
    ? 'Direct Counterparty (1-Hop)'
    : 'Secondary Counterparty (2-Hop)';

  return (
    <div
      style={{
        position: 'absolute',
        top: 14,
        right: 14,
        width: 310,
        backgroundColor: '#ffffff',
        border: '1px solid var(--border-default)',
        borderRadius: 'var(--radius-sm)',
        boxShadow: 'var(--shadow-md)',
        padding: 16,
        zIndex: 25,
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 10 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>
              Account Profile
            </span>
            <span
              style={{
                fontSize: 9.5,
                fontWeight: 600,
                padding: '1px 5px',
                borderRadius: 3,
                backgroundColor: node.isTarget ? '#eff6ff' : 'var(--bg-surface-subtle)',
                color: node.isTarget ? '#1d4ed8' : 'var(--text-secondary)',
                border: node.isTarget ? '1px solid #bfdbfe' : '1px solid var(--border-subtle)',
              }}
            >
              {relationshipLabel}
            </span>
          </div>
          <div className="mono" style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)', marginTop: 4 }}>
            {node.accountId}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 1 }} className="mono">
            Node #{node.nodeId}
          </div>
        </div>
        <button
          onClick={onClose}
          style={{ background: 'transparent', border: 'none', cursor: 'pointer', padding: 2 }}
          title="Close Panel"
        >
          <X size={16} color="var(--text-muted)" />
        </button>
      </div>

      {/* Metrics Grid: Distinguish Risk Level & Risk Score */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 14 }}>
        <div style={{ padding: 8, backgroundColor: 'var(--bg-surface-subtle)', borderRadius: 'var(--radius-xs)' }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Risk Level</div>
          <div style={{ marginTop: 4 }}>
            <RiskBadge level={node.riskLevel} />
          </div>
        </div>

        <div style={{ padding: 8, backgroundColor: 'var(--bg-surface-subtle)', borderRadius: 'var(--radius-xs)' }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Model Risk Score</div>
          <div className="mono tabular" style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginTop: 4 }}>
            {(node.riskScore * 100).toFixed(1)}% <span style={{ fontSize: 10.5, color: 'var(--text-muted)', fontWeight: 500 }}>({node.riskScore.toFixed(3)})</span>
          </div>
        </div>

        <div style={{ padding: 8, backgroundColor: 'var(--bg-surface-subtle)', borderRadius: 'var(--radius-xs)' }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Total Volume</div>
          <div className="mono tabular" style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginTop: 4 }}>
            {formatCurrency(node.totalVolume)}
          </div>
        </div>

        <div style={{ padding: 8, backgroundColor: 'var(--bg-surface-subtle)', borderRadius: 'var(--radius-xs)' }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Transactions</div>
          <div className="tabular" style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', marginTop: 4 }}>
            {node.transactionCount}
          </div>
        </div>
      </div>

      {/* Primary Actions */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <button
          className="btn btn-primary"
          onClick={handleInvestigate}
          style={{ width: '100%', justifyContent: 'center' }}
        >
          <span>Investigate Account</span>
          <ArrowUpRight size={13} />
        </button>

        {!node.isTarget && onCenterAccount && (
          <button
            className="btn btn-secondary"
            onClick={() => onCenterAccount(node.accountId)}
            style={{ width: '100%', justifyContent: 'center' }}
            title="Recenter network on this account"
          >
            <Focus size={13} />
            <span>Recenter Graph on Node</span>
          </button>
        )}
      </div>
    </div>
  );
};
