import React, { useEffect, useState } from 'react';
import { Report } from '../types';
import { reportService } from '../services/reportService';
import { PageHeader } from '../components/common/PageHeader';
import { ReportList } from '../components/reports/ReportList';
import { LoadingState } from '../components/common/LoadingState';
import { EmptyState } from '../components/common/EmptyState';
import { ErrorState } from '../components/common/ErrorState';

export const ReportsPage: React.FC = () => {
  const [reports, setReports] = useState<Report[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadReports = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await reportService.getReports();
      setReports(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load filed reports');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadReports();
  }, []);

  return (
    <div>
      <PageHeader
        title="Reports"
        subtitle="Formal compliance filings, audit memos, and investigation adjudication records"
      />

      {isLoading ? (
        <LoadingState message="Retrieving compliance reports..." />
      ) : error ? (
        <ErrorState message={error} onRetry={loadReports} />
      ) : reports.length === 0 ? (
        <EmptyState
          title="No Reports Filed"
          description="No formal investigation reports have been generated yet. Open an investigation to file a report."
        />
      ) : (
        <ReportList reports={reports} />
      )}
    </div>
  );
};
