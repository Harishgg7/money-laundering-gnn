import { RiskLevel } from './account';

export type InvestigatorDecision =
  | 'Continue Surveillance'
  | 'Escalate for Further Review'
  | 'File Investigation Report'
  | 'File Regulatory Report'
  | 'Close — No Further Action';

export type ReportStatus = 'Filed' | 'Draft' | 'Submitted' | 'Under Review';

export interface ReportTransactionEvidence {
  totalTransferred: number;
  transactionCount: number;
  connectedAccountsCount: number;
}

export interface ReportModelAssessment {
  riskScore: number;
  threshold: number;
  assessmentNote: string;
}

export interface Report {
  id: string; // "REP-2026-001"
  caseId: string;
  accountId: string;
  riskLevel: RiskLevel;
  decision: InvestigatorDecision;
  status: ReportStatus;
  filedDate: string;
  investigatorName: string;
  summary: string;
  observedPattern: string;
  evidence: ReportTransactionEvidence;
  modelAssessment: ReportModelAssessment;
  notes: string;
  createdAt: string;
  updatedAt?: string;
}

export interface CreateReportInput {
  id?: string;
  caseId: string;
  accountId: string;
  riskLevel: RiskLevel;
  decision: InvestigatorDecision;
  summary: string;
  observedPattern: string;
  evidence: ReportTransactionEvidence;
  modelAssessment: ReportModelAssessment;
  notes: string;
  investigatorName?: string;
}

