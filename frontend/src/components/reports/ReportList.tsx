import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Report } from '../../types';
import { RiskBadge } from '../common/RiskBadge';
import { StatusBadge } from '../common/StatusBadge';
import { ArrowUpRight, FilePlus } from 'lucide-react';

interface ReportListProps {
  reports: Report[];
}

export const ReportList: React.FC<ReportListProps> = ({ reports }) => {
  const navigate = useNavigate();

  return (
    <div className="table-card">
      <div className="table-toolbar">
        <div>
          <span className="table-title">Compliance Reports Registry</span>
          <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 8 }}>
            Formal investigation reports, regulatory filings, and adjudication decisions
          </span>
        </div>
        <button
          className="btn btn-primary btn-sm"
          onClick={() => navigate('/investigations')}
        >
          <FilePlus size={12} />
          <span>New Report from Case</span>
        </button>
      </div>

      <table className="trace-table">
        <thead>
          <tr>
            <th>Report ID</th>
            <th>Case ID</th>
            <th>Account</th>
            <th>Risk Level</th>
            <th>Investigator Decision</th>
            <th>Filed Date</th>
            <th>Status</th>
            <th style={{ textAlign: 'right' }}>Action</th>
          </tr>
        </thead>
        <tbody>
          {reports.map((rep) => (
            <tr key={rep.id}>
              <td>
                <span className="mono" style={{ fontWeight: 600, color: 'var(--primary-blue)' }}>
                  {rep.id}
                </span>
              </td>
              <td>
                <span className="mono" style={{ color: 'var(--text-secondary)' }}>
                  {rep.caseId}
                </span>
              </td>
              <td>
                <span className="mono" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                  {rep.accountId}
                </span>
              </td>
              <td>
                <RiskBadge level={rep.riskLevel} />
              </td>
              <td>
                <span style={{ fontSize: 12, fontWeight: 500, color: 'var(--text-primary)' }}>
                  {rep.decision}
                </span>
              </td>
              <td className="mono tabular" style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                {rep.filedDate}
              </td>
              <td>
                <StatusBadge status={rep.status} />
              </td>
              <td style={{ textAlign: 'right' }}>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => navigate(`/reports/${rep.id}`)}
                  title="View complete report details"
                >
                  <span>View</span>
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
