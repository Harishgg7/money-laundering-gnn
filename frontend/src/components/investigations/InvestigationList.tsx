import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Investigation } from '../../types';
import { RiskBadge } from '../common/RiskBadge';
import { StatusBadge } from '../common/StatusBadge';
import { ArrowUpRight } from 'lucide-react';
import { formatAmount } from '../../utils/formatters';

interface InvestigationListProps {
  investigations: Investigation[];
}

export const InvestigationList: React.FC<InvestigationListProps> = ({ investigations }) => {
  const navigate = useNavigate();

  return (
    <div className="table-card">
      <div className="table-toolbar">
        <div>
          <span className="table-title">Suspicious Activity Queue</span>
          <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 8 }}>
            Flagged accounts requiring investigator review and adjudication
          </span>
        </div>
      </div>

      <table className="trace-table">
        <thead>
          <tr>
            <th>Case ID</th>
            <th>Account</th>
            <th>Risk Level</th>
            <th style={{ textAlign: 'right' }}>Amount</th>
            <th style={{ textAlign: 'right' }}>Transactions</th>
            <th>Observed Pattern</th>
            <th>Status</th>
            <th style={{ textAlign: 'right' }}>Action</th>
          </tr>
        </thead>
        <tbody>
          {investigations.map((inv) => (
            <tr key={inv.id}>
              <td>
                <span className="mono" style={{ fontWeight: 600, color: 'var(--primary-blue)' }}>
                  {inv.caseId}
                </span>
              </td>
              <td>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span className="mono" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                    {inv.accountId}
                  </span>
                  <span style={{ fontSize: 10.5, color: 'var(--text-muted)' }} className="mono">
                    Node #{inv.nodeId}
                  </span>
                </div>
              </td>
              <td>
                <RiskBadge level={inv.riskLevel} score={inv.riskScore} showScore={true} />
              </td>
              <td style={{ textAlign: 'right' }} className="tabular">
                <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                  {formatAmount(inv.totalVolume)}
                </span>
              </td>
              <td style={{ textAlign: 'right' }} className="tabular">
                {inv.transactionCount}
              </td>
              <td>
                <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                  {inv.primaryPattern}
                </span>
              </td>
              <td>
                <StatusBadge status={inv.status} />
              </td>
              <td style={{ textAlign: 'right' }}>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => navigate(`/investigations/${inv.caseId}`)}
                  title="Open investigation detail"
                >
                  <span>Open Case</span>
                  <ArrowUpRight size={12} />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
