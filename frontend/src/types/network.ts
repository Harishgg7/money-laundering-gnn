import { RiskLevel } from './account';

export interface NetworkNode {
  id: string; // usually accountId or string id
  nodeId: number;
  accountId: string;
  label: string;
  riskScore: number;
  riskLevel: RiskLevel;
  totalVolume: number;
  transactionCount: number;
  isTarget?: boolean;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
}

export interface NetworkEdge {
  id: string;
  source: string; // node id
  target: string; // node id
  sourceLabel: string;
  targetLabel: string;
  amount: number;
  importance: number; // 0.0 to 1.0 (GNNExplainer or flow importance)
  transactionCount?: number;
}

export interface NetworkGraphData {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
}

export interface NetworkFilter {
  searchQuery: string;
  riskLevel: 'ALL' | RiskLevel;
  transactionType?: string;
  timeRange?: string;
}
