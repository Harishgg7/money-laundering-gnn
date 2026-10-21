import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Investigation, InvestigationStatus } from '../types';
import { investigationService } from '../services/investigationService';
import { InvestigationDetail } from '../components/investigations/InvestigationDetail';
import { LoadingState } from '../components/common/LoadingState';
import { EmptyState } from '../components/common/EmptyState';
import { ErrorState } from '../components/common/ErrorState';

export const InvestigationDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [investigation, setInvestigation] = useState<Investigation | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadCase = async (caseId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await investigationService.getInvestigationById(caseId);
      setInvestigation(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load case dossier');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (id) {
      loadCase(id);
    }
  }, [id]);

  const handleUpdateStatus = async (newStatus: InvestigationStatus) => {
    if (!investigation) return;
    const success = await investigationService.updateStatus(investigation.id, newStatus);
    if (success) {
      setInvestigation((prev) => (prev ? { ...prev, status: newStatus } : null));
    }
  };

  if (isLoading) {
    return <LoadingState message="Assembling investigation dossier and local graph..." />;
  }

  if (error) {
    return <ErrorState message={error} onRetry={() => id && loadCase(id)} />;
  }

  if (!investigation) {
    return (
      <EmptyState
        title="Case Not Found"
        description={`The requested investigation '${id}' does not exist or has been archived.`}
        action={
          <button className="btn btn-secondary btn-sm" onClick={() => navigate('/investigations')}>
            Return to Investigations
          </button>
        }
      />
    );
  }

  return (
    <InvestigationDetail
      investigation={investigation}
      onUpdateStatus={handleUpdateStatus}
    />
  );
};
