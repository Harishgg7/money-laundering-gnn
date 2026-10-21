import React from 'react';

interface ErrorStateProps {
  message?: string;
  onRetry?: () => void;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  message = 'Unable to load information.',
  onRetry,
}) => {
  return (
    <div className="feedback-box">
      <div className="feedback-title" style={{ color: 'var(--risk-high-text)' }}>
        Operation Failed
      </div>
      <div className="feedback-desc">{message}</div>
      {onRetry && (
        <button
          className="btn btn-secondary btn-sm"
          onClick={onRetry}
          style={{ marginTop: 12 }}
        >
          Try Again
        </button>
      )}
    </div>
  );
};
