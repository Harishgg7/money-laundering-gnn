import { RiskLevel } from './account';

export type TimeRangeFilter = '7D' | '30D' | '3M' | '1Y' | 'CUSTOM';

export interface TopAccountSummary {
  accountId: string;
  nodeId: number;
  transactionValue: number;
  transactionCount: number;
  riskLevel: RiskLevel;
  riskScore: number;
  caseId?: string;
  primaryPattern?: string;
}

export interface ActivityDataPoint {
  date: string;
  label: string;
  volume: number; // In currency units
  transactionCount: number;
  flaggedVolume: number;
}

export interface CategoryDistribution {
  category: string;
  percentage: number;
  value: number;
}

export interface DashboardMetrics {
  totalTransactionValue: number;
  totalTransactions: number;
  activeAccounts: number;
  suspiciousAccounts: number;
  timeRange: TimeRangeFilter;
  activityOverTime: ActivityDataPoint[];
  categoryDistribution: CategoryDistribution[];
  topAccounts: TopAccountSummary[];
}
