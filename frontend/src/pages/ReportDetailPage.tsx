import React, { useEffect, useState } from 'react';
import { useParams, useSearchParams, useNavigate } from 'react-router-dom';
import { Report } from '../types';
import { reportService } from '../services/reportService';
import { investigationService } from '../services/investigationService';
import { ReportDetail } from '../components/reports/ReportDetail';
import { LoadingState } from '../components/common/LoadingState';
import { EmptyState } from '../components/common/EmptyState';
import { ErrorState } from '../components/common/ErrorState';

export const ReportDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [searchParams] = useSearchParams();
  const caseOrAccountParam = searchParams.get('case') || searchParams.get('account');
  const navigate = useNavigate();

  const [report, setReport] = useState<Report | null>(null);
  const [casePreload, setCasePreload] = useState<{
    caseId: string;
    accountId: string;
    riskLevel: 'HIGH' | 'MEDIUM' | 'LOW';
    riskScore: number;
    summary: string;
    primaryPattern: string;
    totalVolume: number;
    transactionCount: number;
    connectedAccountsCount: number;
  } | null>(null);

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true);
      setError(null);
      try {
        if (id && id !== 'new') {
          // Existing report
          const existing = await reportService.getReportById(id);
          setReport(existing);
        } else if (caseOrAccountParam) {
          // New report filing pre-populated dynamically from case or account
          const inv = await investigationService.getInvestigationById(caseOrAccountParam);
          if (inv) {
            setCasePreload({
              caseId: inv.caseId || `CASE-${inv.accountId}`,
              accountId: inv.accountId,
              riskLevel: inv.riskLevel,
              riskScore: inv.riskScore,
              summary: inv.summary || `Automated compliance investigation opened for subject account ${inv.accountId}.`,
              primaryPattern: inv.primaryPattern || 'Layering',
              totalVolume: inv.totalVolume || 0,
              transactionCount: inv.transactionCount || 0,
              connectedAccountsCount: inv.connectedAccountsCount || 0,
            });
          }
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unable to load report document');
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [id, caseOrAccountParam]);

  if (isLoading) {
    return <LoadingState message="Loading compliance report document..." />;
  }

  if (error) {
    return <ErrorState message={error} onRetry={() => window.location.reload()} />;
  }

  // If looking for an existing report that doesn't exist
  if (!report && id && id !== 'new') {
    return (
      <EmptyState
        title="Report Not Found"
        description={`The requested report '${id}' could not be located in the compliance registry.`}
        action={
          <button className="btn btn-secondary btn-sm" onClick={() => navigate('/reports')}>
            Return to Reports
          </button>
        }
      />
    );
  }

  // If visiting /reports/new without any case or account parameter
  if (!report && !casePreload && (id === 'new' || !id)) {
    return (
      <EmptyState
        title="Select Case to File Report"
        description="A compliance report must be filed from an active investigation dossier. Please select a case from the queue."
        action={
          <button className="btn btn-primary btn-sm" onClick={() => navigate('/investigations')}>
            Go to Investigations
          </button>
        }
      />
    );
  }

  return (
    <ReportDetail
      initialReport={report || undefined}
      casePreload={casePreload || undefined}
    />
  );
};
