import { NetworkGraphData, NetworkFilter, NetworkNode, RiskLevel } from '../types';
import { mockNetworkGraph } from '../data/mock/networkData';
import { request } from './apiClient';

export const networkService = {
  async getGraph(filter?: Partial<NetworkFilter>): Promise<NetworkGraphData> {
    const q = filter?.searchQuery ? filter.searchQuery.trim() : '';
    const risk = filter?.riskLevel || 'ALL';
    const url = q
      ? `/api/network?account=${encodeURIComponent(q)}&risk=${risk}`
      : `/api/network?risk=${risk}`;

    return request<NetworkGraphData>(
      url,
      undefined,
      () => mockNetworkGraph
    );
  },

  async searchAccounts(query: string, limit = 20): Promise<Array<{
    accountId: string;
    nodeId: number;
    riskScore: number;
    riskLevel: RiskLevel;
    caseId?: string;
    title?: string;
  }>> {
    return request(
      `/api/network/search?q=${encodeURIComponent(query)}&limit=${limit}`
    );
  },

  async getNodeById(idOrAccountId: string): Promise<NetworkNode | null> {
    return request<NetworkNode | null>(
      `/api/accounts/${encodeURIComponent(idOrAccountId)}`,
      undefined,
      () => {
        const found = mockNetworkGraph.nodes.find(
          (n) => n.id === idOrAccountId || n.accountId === idOrAccountId || n.nodeId.toString() === idOrAccountId
        );
        return found || null;
      }
    );
  },
};
