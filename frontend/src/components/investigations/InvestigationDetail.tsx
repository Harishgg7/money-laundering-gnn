import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Investigation, InvestigationStatus } from '../../types';
import { RiskBadge } from '../common/RiskBadge';
import { StatusBadge } from '../common/StatusBadge';
import { NetworkGraph } from '../network/NetworkGraph';
import { PatternCard } from './PatternCard';
import { ArrowLeft, ExternalLink, FileText, Eye, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import { formatAmount } from '../../utils/formatters';

interface InvestigationDetailProps {
  investigation: Investigation;
  onUpdateStatus: (newStatus: InvestigationStatus) => Promise<boolean | void>;
}

export const InvestigationDetail: React.FC<InvestigationDetailProps> = ({
  investigation,
  onUpdateStatus,
}) => {
  const navigate = useNavigate();
  const [isUpdating, setIsUpdating] = useState(false);
  const [updateError, setUpdateError] = useState<string | null>(null);

  const handleStatusChange = async (status: InvestigationStatus) => {
    setIsUpdating(true);
    setUpdateError(null);
    try {
      const res = await onUpdateStatus(status);
      if (res === false) {
        setUpdateError('Failed to update case status. Please try again.');
      }
    } catch (err: any) {
      setUpdateError(err?.message || 'Failed to update case status. Please try again.');
    } finally {
      setIsUpdating(false);
    }
  };

  const handleFileReport = () => {
    // Navigate directly to create/open report with case pre-population
    navigate(`/reports/new?case=${encodeURIComponent(investigation.caseId)}&account=${encodeURIComponent(investigation.accountId)}`);
  };


  return (
    <div>
      {/* Top Breadcrumb & Action Bar */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
        <button
          className="btn btn-secondary btn-sm"
          onClick={() => navigate('/investigations')}
        >
          <ArrowLeft size={13} />
          <span>Back to Investigations</span>
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>Status:</span>
          <StatusBadge status={investigation.status} />
        </div>
      </div>

      {/* Case Header Strip */}
      <div
        className="panel-card"
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 12,
          padding: '10px 16px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
          <div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Case Reference
            </div>
            <div className="mono" style={{ fontSize: 15, fontWeight: 700, color: 'var(--text-primary)', marginTop: 1 }}>
              {investigation.caseId}
            </div>
          </div>
          <div style={{ height: 24, width: 1, backgroundColor: 'var(--border-subtle)' }} />
          <div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Subject Account
            </div>
            <div className="mono" style={{ fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', marginTop: 1 }}>
              {investigation.accountId}
            </div>
          </div>
          <div style={{ height: 24, width: 1, backgroundColor: 'var(--border-subtle)' }} />
          <div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Risk Level
            </div>
            <div style={{ marginTop: 2 }}>
              <RiskBadge level={investigation.riskLevel} />
            </div>
          </div>
          <div style={{ height: 24, width: 1, backgroundColor: 'var(--border-subtle)' }} />
          <div>
            <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Risk Score
            </div>
            <div className="mono tabular" style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginTop: 2 }}>
              {(investigation.riskScore * 100).toFixed(0)}% <span style={{ fontSize: 11, fontWeight: 500, color: 'var(--text-muted)' }}>({investigation.riskScore.toFixed(2)})</span>
            </div>
          </div>
        </div>

        {/* Primary Action Button in Header */}
        <button className="btn btn-primary btn-sm" onClick={handleFileReport}>
          <FileText size={13} />
          <span>File Investigation Report</span>
        </button>
      </div>

      {/* Main Two-Column Workflow Workspace */}
      <div className="investigation-split" style={{ gap: 14 }}>
        {/* ================= LEFT COLUMN: NETWORK GRAPH ================= */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          <div className="panel-card" style={{ padding: 0, overflow: 'hidden' }}>
            <div
              style={{
                padding: '8px 12px',
                borderBottom: '1px solid var(--border-subtle)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                backgroundColor: 'var(--bg-surface)',
              }}
            >
              <div>
                <span style={{ fontWeight: 600, fontSize: 12, color: 'var(--text-primary)' }}>
                  Local Transaction Neighborhood
                </span>
                <span style={{ fontSize: 11, color: 'var(--text-muted)', marginLeft: 6 }}>
                  (Target: {investigation.accountId})
                </span>
              </div>
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => navigate(`/network?account=${investigation.accountId}`)}
                title="Open full interactive network map"
              >
                <span>View Full Network</span>
                <ExternalLink size={11} />
              </button>
            </div>

            <div style={{ height: 430, width: '100%' }}>
              <NetworkGraph
                data={investigation.subgraph}
                selectedNodeId={investigation.accountId}
                targetAccountId={investigation.accountId}
                onSelectNode={(node) => {
                  if (node && node.accountId !== investigation.accountId) {
                    navigate(`/network?account=${node.accountId}`);
                  }
                }}
                height={430}
              />
            </div>
          </div>

          {/* Quick Guidance Alert Box */}
          <div
            style={{
              padding: '8px 12px',
              backgroundColor: 'var(--bg-surface-subtle)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-sm)',
              display: 'flex',
              alignItems: 'flex-start',
              gap: 8,
              fontSize: 11,
              color: 'var(--text-secondary)',
            }}
          >
            <Info size={13} color="var(--primary-blue)" style={{ flexShrink: 0, marginTop: 1 }} />
            <span>
              The local graph highlights direct transaction flows involving account <b>{investigation.accountId}</b>. 
              Clicking <b>View Full Network</b> navigates to the global interactive network map.
            </span>
          </div>
        </div>

        {/* ================= RIGHT COLUMN: INVESTIGATION SUMMARY ================= */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {/* WHO & WHAT: Account Summary Strip */}
          <div className="panel-card" style={{ padding: '12px 14px' }}>
            <div className="panel-title" style={{ marginBottom: 8, fontSize: 12.5 }}>
              <span>Account &amp; Activity Summary</span>
              <span style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>WHO &amp; WHAT</span>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8, marginBottom: 8 }}>
              <div style={{ padding: '6px 8px', backgroundColor: 'var(--bg-surface-subtle)', borderRadius: 'var(--radius-xs)' }}>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Total Volume</div>
                <div className="mono tabular" style={{ fontSize: 12.5, fontWeight: 700, color: 'var(--text-primary)', marginTop: 2 }}>
                  {formatAmount(investigation.totalVolume)}
                </div>
              </div>

              <div style={{ padding: '6px 8px', backgroundColor: 'var(--bg-surface-subtle)', borderRadius: 'var(--radius-xs)' }}>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Transactions</div>
                <div className="tabular" style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-primary)', marginTop: 2 }}>
                  {investigation.transactionCount}
                </div>
              </div>

              <div style={{ padding: '6px 8px', backgroundColor: 'var(--bg-surface-subtle)', borderRadius: 'var(--radius-xs)' }}>
                <div style={{ fontSize: 10, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Counterparties</div>
                <div className="tabular" style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-primary)', marginTop: 2 }}>
                  {investigation.connectedAccountsCount} accounts
                </div>
              </div>
            </div>

            <div style={{ fontSize: 11.5, color: 'var(--text-secondary)', lineHeight: 1.4 }}>
              {investigation.summary}
            </div>
          </div>

          {/* WHY: Plain English Explanatory Reasons */}
          <div className="panel-card" style={{ padding: '12px 14px' }}>
            <div className="panel-title" style={{ marginBottom: 6, fontSize: 12.5 }}>
              <span>Why was this account flagged?</span>
              <span style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>WHY</span>
            </div>

            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
              Activity and relational flow patterns contributing to the model's risk assessment:
            </div>

            <ul className="evidence-list" style={{ gap: 6 }}>
              {investigation.whyFlagged.map((reason, idx) => (
                <li key={idx} className="evidence-item" style={{ fontSize: 12, lineHeight: 1.4 }}>
                  <span className="evidence-bullet" style={{ marginTop: 5 }} />
                  <span>{reason}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Observed Patterns Component */}
          <PatternCard patterns={investigation.patterns} />

          {/* WHAT NEXT: Investigator Actions Bar */}
          <div className="panel-card" style={{ padding: '12px 14px' }}>
            <div className="panel-title" style={{ marginBottom: 8, fontSize: 12.5 }}>
              <span>Investigator Decision Workflow</span>
              <span style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>WHAT NEXT</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {/* Primary Workflow Action */}
              <button
                className="btn btn-primary"
                onClick={handleFileReport}
                style={{ padding: '7px 12px', fontSize: 12, fontWeight: 600 }}
              >
                <FileText size={13} />
                <span>File Investigation Report</span>
              </button>

              {/* Secondary Adjudication Actions */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6, marginTop: 2 }}>
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => handleStatusChange('Under Surveillance')}
                  disabled={isUpdating}
                  title="Place under continuous 30-day monitoring"
                >
                  <Eye size={12} />
                  <span>Surveillance</span>
                </button>

                <button
                  className="btn btn-danger btn-sm"
                  onClick={() => handleStatusChange('Escalated')}
                  disabled={isUpdating}
                  title="Escalate case to senior compliance officer"
                >
                  <AlertTriangle size={12} />
                  <span>Escalate</span>
                </button>

                <button
                  className="btn btn-secondary btn-sm"
                  onClick={() => handleStatusChange('Closed — No Further Action')}
                  disabled={isUpdating}
                  title="Close investigation with no further action"
                >
                  <CheckCircle size={12} />
                  <span>Close</span>
                </button>
              </div>

              {updateError && (
                <div style={{ color: 'var(--risk-high, #ef4444)', fontSize: 11.5, marginTop: 4 }}>
                  {updateError}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
