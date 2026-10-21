import React, { useEffect, useState, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { NetworkGraphData, NetworkNode, RiskLevel } from '../types';
import { networkService } from '../services/networkService';
import { PageHeader } from '../components/common/PageHeader';
import { NetworkGraph } from '../components/network/NetworkGraph';
import { GraphControls } from '../components/network/GraphControls';
import { AccountPanel } from '../components/network/AccountPanel';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorState } from '../components/common/ErrorState';

export const NetworkPage: React.FC = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const urlAccount = searchParams.get('account') || searchParams.get('search') || '';

  const [activeAccount, setActiveAccount] = useState<string>(urlAccount);
  const [searchQuery, setSearchQuery] = useState<string>(urlAccount);
  const [graphData, setGraphData] = useState<NetworkGraphData | null>(null);
  const [selectedNode, setSelectedNode] = useState<NetworkNode | null>(null);
  const [riskFilter, setRiskFilter] = useState<'ALL' | RiskLevel>('ALL');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Sync state if URL query parameter changes externally
  useEffect(() => {
    setActiveAccount(urlAccount);
    setSearchQuery(urlAccount);
  }, [urlAccount]);

  // Load bounded network neighborhood
  const loadGraph = useCallback(async (account: string, risk: 'ALL' | RiskLevel) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await networkService.getGraph({ searchQuery: account, riskLevel: risk });
      setGraphData(data);

      // Identify and auto-select canonical target node
      const targetQuery = account.trim();
      let targetNode: NetworkNode | undefined;
      if (targetQuery) {
        targetNode = data.nodes.find(
          (n) => n.accountId.toLowerCase() === targetQuery.toLowerCase() || n.isTarget
        );
      }
      if (!targetNode) {
        targetNode = data.nodes.find((n) => n.isTarget) || data.nodes[0];
      }
      if (targetNode) {
        setSelectedNode(targetNode);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to render network graph');
      setGraphData(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Fetch only when committed active account or risk filter changes
  useEffect(() => {
    loadGraph(activeAccount, riskFilter);
  }, [activeAccount, riskFilter, loadGraph]);

  const handleSelectAccount = (accId: string) => {
    const trimmed = accId.trim();
    setSearchQuery(trimmed);
    setActiveAccount(trimmed);
    if (trimmed) {
      setSearchParams({ account: trimmed });
    } else {
      setSearchParams({});
    }
  };

  const handleReset = () => {
    setSearchQuery('');
    setActiveAccount('');
    setRiskFilter('ALL');
    setSelectedNode(null);
    setSearchParams({});
  };

  return (
    <div>
      <PageHeader
        title="Transaction Network Explorer"
        subtitle="Account relationship graph and multi-hop fund flow topology"
      />

      <GraphControls
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        onSelectAccount={handleSelectAccount}
        riskFilter={riskFilter}
        onRiskFilterChange={(newRisk) => {
          setRiskFilter(newRisk);
          if (selectedNode && newRisk !== 'ALL' && selectedNode.riskLevel !== newRisk) {
            setSelectedNode(null);
          }
        }}
        onReset={handleReset}
      />

      {isLoading && !graphData ? (
        <LoadingState message="Extracting network topology neighborhood..." />
      ) : error ? (
        <ErrorState
          message={error}
          onRetry={handleReset}
        />
      ) : graphData ? (
        <div style={{ position: 'relative', height: 'calc(100vh - 210px)', minHeight: 480 }}>
          <NetworkGraph
            data={graphData}
            selectedNodeId={selectedNode?.id || null}
            onSelectNode={setSelectedNode}
            targetAccountId={activeAccount || null}
            height="100%"
          />

          {/* Node Detail Slide-Over Card */}
          <AccountPanel
            node={selectedNode}
            onClose={() => setSelectedNode(null)}
            onCenterAccount={handleSelectAccount}
          />
        </div>
      ) : null}
    </div>
  );
};
