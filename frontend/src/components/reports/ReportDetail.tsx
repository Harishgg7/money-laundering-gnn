import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Report, InvestigatorDecision, CreateReportInput } from '../../types';
import { RiskBadge } from '../common/RiskBadge';
import { StatusBadge } from '../common/StatusBadge';
import { ArrowLeft, Download, Mail, Save, AlertCircle, CheckCircle2 } from 'lucide-react';
import { reportService } from '../../services/reportService';

interface ReportDetailProps {
  initialReport?: Report;
  casePreload?: {
    caseId: string;
    accountId: string;
    riskLevel: 'HIGH' | 'MEDIUM' | 'LOW';
    riskScore: number;
    summary: string;
    primaryPattern: string;
    totalVolume: number;
    transactionCount: number;
    connectedAccountsCount: number;
  };
}

export const ReportDetail: React.FC<ReportDetailProps> = ({
  initialReport,
  casePreload,
}) => {
  const navigate = useNavigate();

  // If new report from case, pre-populate; otherwise use existing report values
  const [decision, setDecision] = useState<InvestigatorDecision>(
    initialReport?.decision || 'Escalate for Further Review'
  );
  const [notes, setNotes] = useState<string>(
    initialReport?.notes || ''
  );
  const [isSaving, setIsSaving] = useState(false);
  const [pdfStatus, setPdfStatus] = useState<'ready' | 'generating' | 'downloaded' | 'failed'>('ready');
  const [emailStatus, setEmailStatus] = useState<'ready' | 'sending' | 'sent' | 'failed' | 'unconfigured'>('ready');
  const [emailRecipient, setEmailRecipient] = useState<string>('compliance-fiu@bank.internal');
  const [showEmailInput, setShowEmailInput] = useState<boolean>(false);
  const [boundaryNotification, setBoundaryNotification] = useState<{
    type: 'info' | 'warning' | 'success';
    message: string;
  } | null>(null);

  const caseId = initialReport?.caseId ?? casePreload?.caseId ?? 'CASE-DRAFT';
  const accountId = initialReport?.accountId ?? casePreload?.accountId ?? 'UNKNOWN';
  const riskLevel = initialReport?.riskLevel ?? casePreload?.riskLevel ?? 'MEDIUM';
  const riskScore = initialReport?.modelAssessment?.riskScore ?? casePreload?.riskScore ?? 0.50;
  const summary = initialReport?.summary ?? casePreload?.summary ?? (casePreload?.accountId ? `Automated compliance investigation opened for subject account ${casePreload.accountId}.` : 'Investigation opened for compliance adjudication.');
  const pattern = initialReport?.observedPattern ?? casePreload?.primaryPattern ?? 'Layering';
  const evidence = initialReport?.evidence ?? {
    totalTransferred: casePreload?.totalVolume ?? 0,
    transactionCount: casePreload?.transactionCount ?? 0,
    connectedAccountsCount: casePreload?.connectedAccountsCount ?? 0,
  };


  const formatAmount = (val: number) => {
    if (val >= 10000000) return `₹${(val / 10000000).toFixed(2)} Cr`;
    if (val >= 100000) return `₹${(val / 100000).toFixed(2)} L`;
    return `$${val.toLocaleString()}`;
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      const input: CreateReportInput = {
        id: initialReport?.id,
        caseId,
        accountId,
        riskLevel,
        decision,
        summary,
        observedPattern: pattern,
        evidence,
        modelAssessment: {
          riskScore,
          threshold: 0.3781,
          assessmentNote: 'Calculated via 3-hop GraphSAGE inductive message passing.',
        },
        notes,
      };
      const saved = await reportService.saveReport(input);
      setBoundaryNotification({
        type: 'success',
        message: `Report ${saved.id} successfully persisted with decision '${decision}'.`,
      });
      if (!initialReport?.id) {
        setTimeout(() => navigate(`/reports/${saved.id}`), 800);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unable to save report';
      setBoundaryNotification({
        type: 'warning',
        message: `Unable to save report: ${msg}`,
      });
    } finally {
      setIsSaving(false);
    }
  };

  const handleDownloadPdf = async () => {
    const reportId = initialReport?.id;
    if (!reportId) {
      setBoundaryNotification({
        type: 'warning',
        message: 'Please save the draft report to the compliance registry first before exporting official PDF.',
      });
      return;
    }

    setPdfStatus('generating');
    try {
      const res = await reportService.downloadPdf(reportId);
      setPdfStatus('downloaded');
      setBoundaryNotification({
        type: 'success',
        message: `Official compliance report PDF (${res.filename}) generated and downloaded successfully.`,
      });
      setTimeout(() => setPdfStatus('ready'), 3500);
    } catch (err) {
      setPdfStatus('failed');
      const msg = err instanceof Error ? err.message : 'PDF generation failed';
      setBoundaryNotification({
        type: 'warning',
        message: `PDF generation failed: ${msg}`,
      });
      setTimeout(() => setPdfStatus('ready'), 4000);
    }
  };

  const handleSendEmail = async () => {
    const reportId = initialReport?.id;
    if (!reportId) {
      setBoundaryNotification({
        type: 'warning',
        message: 'Please save the draft report to the compliance registry first before dispatching via email.',
      });
      return;
    }

    if (!showEmailInput) {
      setShowEmailInput(true);
      return;
    }

    if (!emailRecipient || !emailRecipient.includes('@')) {
      setBoundaryNotification({
        type: 'warning',
        message: 'Invalid recipient email address.',
      });
      return;
    }

    setEmailStatus('sending');
    try {
      const res = await reportService.sendEmail(reportId, emailRecipient);
      if (res.success) {
        setEmailStatus('sent');
        setBoundaryNotification({
          type: 'success',
          message: res.message,
        });
        setShowEmailInput(false);
        setTimeout(() => setEmailStatus('ready'), 3500);
      } else if (res.status === 'smtp_unconfigured') {
        setEmailStatus('unconfigured');
        setBoundaryNotification({
          type: 'info',
          message: res.message,
        });
        setTimeout(() => setEmailStatus('ready'), 4500);
      } else {
        setEmailStatus('failed');
        setBoundaryNotification({
          type: 'warning',
          message: res.message,
        });
        setTimeout(() => setEmailStatus('ready'), 4000);
      }
    } catch (err) {
      setEmailStatus('failed');
      const msg = err instanceof Error ? err.message : 'Email delivery failed';
      setBoundaryNotification({
        type: 'warning',
        message: `Email delivery failed: ${msg}`,
      });
      setTimeout(() => setEmailStatus('ready'), 4000);
    }
  };

  return (
    <div style={{ maxWidth: 840, margin: '0 auto' }}>
      {/* Navigation Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
        <button className="btn btn-secondary btn-sm" onClick={() => navigate('/reports')}>
          <ArrowLeft size={13} />
          <span>Back to Reports</span>
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {initialReport && (
            <>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Status:</span>
              <StatusBadge status={initialReport.status} />
            </>
          )}
        </div>
      </div>

      {/* Service Boundary Notification Alert */}
      {boundaryNotification && (
        <div
          style={{
            padding: '10px 14px',
            marginBottom: 16,
            borderRadius: 'var(--radius-sm)',
            border: `1px solid ${
              boundaryNotification.type === 'success'
                ? 'var(--risk-low-border)'
                : boundaryNotification.type === 'warning'
                ? 'var(--risk-med-border)'
                : 'var(--primary-blue-border)'
            }`,
            backgroundColor:
              boundaryNotification.type === 'success'
                ? 'var(--risk-low-bg)'
                : boundaryNotification.type === 'warning'
                ? 'var(--risk-med-bg)'
                : 'var(--primary-blue-subtle)',
            color:
              boundaryNotification.type === 'success'
                ? 'var(--risk-low-text)'
                : boundaryNotification.type === 'warning'
                ? 'var(--risk-med-text)'
                : 'var(--primary-blue)',
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            fontSize: 12,
          }}
        >
          {boundaryNotification.type === 'success' ? (
            <CheckCircle2 size={15} style={{ flexShrink: 0 }} />
          ) : (
            <AlertCircle size={15} style={{ flexShrink: 0 }} />
          )}
          <span>{boundaryNotification.message}</span>
        </div>
      )}

      {/* Main Printable / Submittable Report Paper Card */}
      <div
        className="panel-card"
        style={{
          padding: 28,
          backgroundColor: '#ffffff',
          border: '1px solid var(--border-default)',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        {/* Report Official Header */}
        <div
          style={{
            borderBottom: '2px solid var(--text-primary)',
            paddingBottom: 14,
            marginBottom: 20,
            display: 'flex',
            alignItems: 'flex-end',
            justifyContent: 'space-between',
          }}
        >
          <div>
            <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              AML Compliance &bull; Financial Crime Intelligence Report
            </div>
            <div style={{ fontSize: 18, fontWeight: 700, color: 'var(--text-primary)', marginTop: 4 }}>
              Investigation Adjudication Dossier
            </div>
          </div>
          <div style={{ textAlign: 'right' }}>
            <div className="mono" style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
              {initialReport?.id || 'NEW DRAFT FILING'}
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
              Date: {initialReport?.filedDate || new Date().toISOString().split('T')[0]}
            </div>
          </div>
        </div>

        {/* Case & Account Information Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14, marginBottom: 20 }}>
          <div style={{ padding: 10, backgroundColor: 'var(--bg-surface-subtle)', borderRadius: 'var(--radius-xs)' }}>
            <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Case ID</div>
            <div className="mono" style={{ fontSize: 13, fontWeight: 700, color: 'var(--primary-blue)', marginTop: 2 }}>
              {caseId}
            </div>
          </div>

          <div style={{ padding: 10, backgroundColor: 'var(--bg-surface-subtle)', borderRadius: 'var(--radius-xs)' }}>
            <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Subject Account</div>
            <div className="mono" style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', marginTop: 2 }}>
              {accountId}
            </div>
          </div>

          <div style={{ padding: 10, backgroundColor: 'var(--bg-surface-subtle)', borderRadius: 'var(--radius-xs)' }}>
            <div style={{ fontSize: 10.5, color: 'var(--text-muted)', textTransform: 'uppercase' }}>Model Risk Assessment</div>
            <div style={{ marginTop: 4, display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
              <RiskBadge level={riskLevel} />
              <span className="mono tabular" style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
                Score: {(riskScore * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>

        {/* Section: Investigation Summary */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 6 }}>
            1. Investigation Summary
          </div>
          <div style={{ fontSize: 12.5, color: 'var(--text-secondary)', lineHeight: 1.5, backgroundColor: 'var(--bg-surface-subtle)', padding: 10, borderRadius: 'var(--radius-xs)' }}>
            {summary}
          </div>
        </div>

        {/* Section: Observed AML Pattern */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 6 }}>
            2. Observed Pattern Classification
          </div>
          <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)' }}>
            {pattern}
          </div>
          <div style={{ fontSize: 11.5, color: 'var(--text-secondary)', marginTop: 2 }}>
            Categorized via multi-hop topological traversal heuristics.
          </div>
        </div>

        {/* Section: Transaction Evidence */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 6 }}>
            3. Quantitative Transaction Evidence
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
            <div style={{ border: '1px solid var(--border-subtle)', padding: 10, borderRadius: 'var(--radius-xs)' }}>
              <div style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>Total Volume Transferred</div>
              <div className="mono tabular" style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', marginTop: 2 }}>
                {formatAmount(evidence.totalTransferred)}
              </div>
            </div>
            <div style={{ border: '1px solid var(--border-subtle)', padding: 10, borderRadius: 'var(--radius-xs)' }}>
              <div style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>Transactions Logged</div>
              <div className="tabular" style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', marginTop: 2 }}>
                {evidence.transactionCount}
              </div>
            </div>
            <div style={{ border: '1px solid var(--border-subtle)', padding: 10, borderRadius: 'var(--radius-xs)' }}>
              <div style={{ fontSize: 10.5, color: 'var(--text-muted)' }}>Connected Accounts</div>
              <div className="tabular" style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', marginTop: 2 }}>
                {evidence.connectedAccountsCount}
              </div>
            </div>
          </div>
        </div>

        {/* Section: Model Assessment */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 6 }}>
            4. Intelligence Engine Assessment
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
            The GraphSAGE relational neural network evaluated this account with a model risk score of <b>{Math.round(riskScore * 100)}%</b> (operating threshold: 0.3781).
            Activity contributing to the model's risk assessment includes multi-counterparty concentration and rapid pass-through velocity.
          </div>
        </div>

        {/* Section: Investigator Decision (Editable) */}
        <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: 20, marginBottom: 20 }}>
          <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: 8 }}>
            5. Compliance Adjudication Decision
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8, marginBottom: 14 }}>
            {(
              [
                'Continue Surveillance',
                'Escalate for Further Review',
                'File Investigation Report',
                'Close — No Further Action',
              ] as InvestigatorDecision[]
            ).map((opt) => (
              <label
                key={opt}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 8,
                  padding: '8px 10px',
                  backgroundColor: decision === opt ? 'var(--primary-blue-subtle)' : 'var(--bg-surface)',
                  border: `1px solid ${decision === opt ? 'var(--primary-blue)' : 'var(--border-subtle)'}`,
                  borderRadius: 'var(--radius-xs)',
                  cursor: 'pointer',
                  fontSize: 12,
                  fontWeight: decision === opt ? 600 : 400,
                  color: decision === opt ? 'var(--primary-blue)' : 'var(--text-primary)',
                }}
              >
                <input
                  type="radio"
                  name="investigatorDecision"
                  value={opt}
                  checked={decision === opt}
                  onChange={() => setDecision(opt)}
                />
                <span>{opt}</span>
              </label>
            ))}
          </div>

          {/* Investigator Findings & Notes */}
          <div>
            <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>
              Investigator Findings / Regulatory Notes
            </label>
            <textarea
              className="input-control"
              rows={4}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Enter detailed compliance findings, subpoena recommendations, or rationale for resolution..."
              style={{ width: '100%', resize: 'vertical', lineHeight: 1.45 }}
            />
          </div>
        </div>

        {/* Investigator Signature Line */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderTop: '1px solid var(--border-subtle)', paddingTop: 14, fontSize: 11, color: 'var(--text-muted)' }}>
          <div>
            Investigator: <b>{initialReport?.investigatorName || 'Harish GG (Lead AML Investigator)'}</b>
          </div>
          <div>
            System: TRACE Compliance v1.0
          </div>
        </div>
      </div>

      {/* Action Buttons Bar */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginTop: 16,
          padding: '12px 18px',
          backgroundColor: 'var(--bg-surface)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-sm)',
        }}
      >
        <button
          className="btn btn-primary"
          onClick={handleSave}
          disabled={isSaving}
          style={{ padding: '8px 18px', fontSize: 12.5 }}
        >
          <Save size={13} />
          <span>{isSaving ? 'Saving...' : 'Save Report'}</span>
        </button>

        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {showEmailInput && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <input
                type="email"
                className="input-control"
                value={emailRecipient}
                onChange={(e) => setEmailRecipient(e.target.value)}
                placeholder="recipient@bank.internal"
                style={{ fontSize: 11.5, padding: '5px 8px', width: 220 }}
              />
              <button
                className="btn btn-primary btn-sm"
                onClick={handleSendEmail}
                disabled={emailStatus === 'sending'}
                style={{ fontSize: 11 }}
              >
                {emailStatus === 'sending' ? 'Sending...' : 'Confirm Send'}
              </button>
              <button
                className="btn btn-secondary btn-sm"
                onClick={() => setShowEmailInput(false)}
                style={{ fontSize: 11 }}
              >
                Cancel
              </button>
            </div>
          )}

          <button
            className="btn btn-secondary btn-sm"
            onClick={handleDownloadPdf}
            disabled={pdfStatus === 'generating'}
            title="Export official investigation compliance PDF document"
          >
            <Download size={13} />
            <span>
              {pdfStatus === 'generating'
                ? 'Generating...'
                : pdfStatus === 'downloaded'
                ? 'Downloaded'
                : pdfStatus === 'failed'
                ? 'Failed'
                : 'Download PDF'}
            </span>
          </button>

          {!showEmailInput && (
            <button
              className="btn btn-secondary btn-sm"
              onClick={handleSendEmail}
              disabled={emailStatus === 'sending'}
              title="Dispatch report to compliance mail relay"
            >
              <Mail size={13} />
              <span>
                {emailStatus === 'sending'
                  ? 'Sending...'
                  : emailStatus === 'sent'
                  ? 'Sent'
                  : emailStatus === 'unconfigured'
                  ? 'Not configured'
                  : emailStatus === 'failed'
                  ? 'Failed'
                  : 'Send via Email'}
              </span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
