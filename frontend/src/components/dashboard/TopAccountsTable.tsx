import React from 'react';
import { useNavigate } from 'react-router-dom';
import { TopAccountSummary } from '../../types';
import { RiskBadge } from '../common/RiskBadge';
import { ArrowUpRight, Share2 } from 'lucide-react';

interface TopAccountsTableProps {
  accounts: TopAccountSummary[];
}

export const TopAccountsTable: React.FC<TopAccountsTableProps> = ({ accounts }) => {
  const navigate = useNavigate();

  const formatAmount = (val: number) => {
    if (val >= 10000000) return `₹${(val / 10000000).toFixed(2)} Cr`;
    if (val >= 100000) return `₹${(val / 100000).toFixed(2)} L`;
    return `$${val.toLocaleString()}`;
  };

  return (
    <div className="table-card">
      <div className="table-toolbar">
        <div>
          <span className="table-title">Top Flagged Accounts</span>
          <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 8 }}>
            Ranked by volume and model topological risk
          </span>
        </div>
      </div>

      <table className="trace-table">
        <thead>
          <tr>
            <th>Account</th>
            <th style={{ textAlign: 'right' }}>Transaction Value</th>
            <th style={{ textAlign: 'right' }}>Transactions</th>
            <th>Observed Pattern</th>
            <th>Risk Level</th>
            <th style={{ textAlign: 'right' }}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {accounts.map((acct) => (
            <tr key={acct.accountId}>
              <td>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <button
                    onClick={() => navigate(`/network?account=${encodeURIComponent(acct.accountId)}`)}
                    className="mono link-hover"
                    style={{
                      background: 'none',
                      border: 'none',
                      padding: 0,
                      textAlign: 'left',
                      fontWeight: 600,
                      color: 'var(--color-primary)',
                      cursor: 'pointer',
                      fontSize: 12.5,
                    }}
                    title="View in Transaction Network Explorer"
                  >
                    {acct.accountId}
                  </button>
                  <span style={{ fontSize: 10.5, color: 'var(--text-muted)' }} className="mono">
                    Node #{acct.nodeId}
                  </span>
                </div>
              </td>
              <td style={{ textAlign: 'right' }} className="tabular">
                <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                  {formatAmount(acct.transactionValue)}
                </span>
              </td>
              <td style={{ textAlign: 'right' }} className="tabular">
                {acct.transactionCount}
              </td>
              <td>
                <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                  {acct.primaryPattern || 'Standard Flow'}
                </span>
              </td>
              <td>
                <RiskBadge level={acct.riskLevel} score={acct.riskScore} showScore={true} />
              </td>
              <td style={{ textAlign: 'right' }}>
                <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => navigate(`/network?account=${encodeURIComponent(acct.accountId)}`)}
                    title="View account relationship network"
                  >
                    <Share2 size={11} />
                    <span>Network</span>
                  </button>
                  <button
                    className="btn btn-primary btn-sm"
                    onClick={() => {
                      navigate(`/investigations/${encodeURIComponent(acct.caseId || acct.accountId)}`);
                    }}
                    title="Open investigation dossier"
                  >
                    <span>Investigate</span>
                    <ArrowUpRight size={11} />
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
