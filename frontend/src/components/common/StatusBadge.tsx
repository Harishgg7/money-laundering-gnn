import React from 'react';
import { InvestigationStatus, ReportStatus } from '../../types';

interface StatusBadgeProps {
  status: InvestigationStatus | ReportStatus | string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status }) => {
  let badgeClass = 'status-new';
  const lower = status.toLowerCase();

  if (lower.includes('review')) {
    badgeClass = 'status-review';
  } else if (lower.includes('surveillance')) {
    badgeClass = 'status-surveillance';
  } else if (lower.includes('escalate')) {
    badgeClass = 'status-escalated';
  } else if (lower.includes('close') || lower.includes('resolve')) {
    badgeClass = 'status-closed';
  } else if (lower.includes('filed')) {
    badgeClass = 'status-surveillance';
  }

  return <span className={`status-pill ${badgeClass}`}>{status}</span>;
};
