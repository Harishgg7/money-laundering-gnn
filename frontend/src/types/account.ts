export type RiskLevel = 'HIGH' | 'MEDIUM' | 'LOW';

export interface Account {
  id: string;
  accountId: string;
  nodeId: number;
  bankId?: string;
  riskScore: number; // 0.0 to 1.0
  riskLevel: RiskLevel;
  totalInflow: number;
  totalOutflow: number;
  transactionCount: number;
  counterpartiesCount: number;
  groundTruth?: number; // 0: Normal, 1: Illicit (for validation context)
  status?: string;
  currency?: string;
}
