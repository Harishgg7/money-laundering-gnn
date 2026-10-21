import React from 'react';

interface EmptyStateProps {
  title: string;
  description: string;
  action?: React.ReactNode;
}

export const EmptyState: React.FC<EmptyStateProps> = ({ title, description, action }) => {
  return (
    <div className="feedback-box">
      <div className="feedback-title">{title}</div>
      <div className="feedback-desc">{description}</div>
      {action && <div style={{ marginTop: 12 }}>{action}</div>}
    </div>
  );
};
