import { RiskLevel } from './account';
import { NetworkGraphData } from './network';
import { Transaction } from './transaction';

export type InvestigationStatus =
  | 'New'
  | 'Under Review'
  | 'Escalated'
  | 'Under Surveillance'
  | 'Resolved'
  | 'Closed — No Further Action';

export type AMLPatternType =
  | 'Layering'
  | 'Structuring'
  | 'Rapid Movement'
  | 'Circular Flow'
  | 'Multiple Counterparties'
  | 'Smurfing / Fan-In'
  | 'Dispersion / Fan-Out';

export interface AMLPattern {
  name: AMLPatternType | string;
  status: 'Observed' | 'Possible' | 'Not observed' | 'Detected' | 'Not detected';
  detail: string;
}

export interface FeatureEvidence {
  feature: string;
  importance: number;
  formattedValue: string;
}

export interface RiskAssessment {
  riskScore: number;
  riskLevel: RiskLevel;
  threshold: number;
  mlpBaselineScore?: number;
  fidelityDrop?: number;
  contributingFactors: string[];
}

export interface Investigation {
  id: string; // e.g. "CASE-2026-0421" or "AML-0241"
  caseId: string;
  accountId: string;
  nodeId: number;
  riskLevel: RiskLevel;
  riskScore: number;
  totalVolume: number;
  transactionCount: number;
  connectedAccountsCount: number;
  primaryPattern: AMLPatternType | string;
  status: InvestigationStatus;
  summary: string;
  whyFlagged: string[];
  patterns: AMLPattern[];
  features?: FeatureEvidence[];
  subgraph: NetworkGraphData;
  recentTransactions?: Transaction[];
  createdAt: string;
  updatedAt: string;
}
