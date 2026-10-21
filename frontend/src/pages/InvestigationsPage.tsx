import React, { useEffect, useState } from 'react';
import { Investigation } from '../types';
import { investigationService } from '../services/investigationService';
import { PageHeader } from '../components/common/PageHeader';
import { InvestigationList } from '../components/investigations/InvestigationList';
import { LoadingState } from '../components/common/LoadingState';
import { EmptyState } from '../components/common/EmptyState';
import { ErrorState } from '../components/common/ErrorState';

export const InvestigationsPage: React.FC = () => {
  const [investigations, setInvestigations] = useState<Investigation[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await investigationService.getInvestigations();
      setInvestigations(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load investigations');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  return (
    <div>
      <PageHeader
        title="Investigations"
        subtitle="Prioritized suspicious activity cases warranting compliance adjudication"
      />

      {isLoading ? (
        <LoadingState message="Loading suspicious case queue..." />
      ) : error ? (
        <ErrorState message={error} onRetry={loadData} />
      ) : investigations.length === 0 ? (
        <EmptyState
          title="No Suspicious Cases Found"
          description="All flagged accounts have been adjudicated or cleared. The system will alert when new topology anomalies are observed."
        />
      ) : (
        <InvestigationList investigations={investigations} />
      )}
    </div>
  );
};
