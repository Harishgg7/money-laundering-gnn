import React, { useEffect, useState } from 'react';
import { DashboardMetrics, TimeRangeFilter } from '../types';
import { dashboardService } from '../services/dashboardService';
import { PageHeader } from '../components/common/PageHeader';
import { MetricCard } from '../components/common/MetricCard';
import { ActivityChart } from '../components/dashboard/ActivityChart';
import { DistributionChart } from '../components/dashboard/DistributionChart';
import { TopAccountsTable } from '../components/dashboard/TopAccountsTable';
import { LoadingState } from '../components/common/LoadingState';
import { ErrorState } from '../components/common/ErrorState';

export const DashboardPage: React.FC = () => {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [timeRange, setTimeRange] = useState<TimeRangeFilter>('30D');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = async (range: TimeRangeFilter) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await dashboardService.getMetrics(range);
      setMetrics(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load dashboard metrics');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics(timeRange);
  }, [timeRange]);

  const formatCurrency = (val: number) => {
    if (val >= 1000000000) return `$${(val / 1000000000).toFixed(2)}B`;
    if (val >= 1000000) return `$${(val / 1000000).toFixed(1)}M`;
    if (val >= 10000000) return `₹${(val / 10000000).toFixed(2)}Cr`;
    if (val >= 100000) return `₹${(val / 100000).toFixed(2)}L`;
    return `$${val.toLocaleString()}`;
  };

  return (
    <div>
      <PageHeader
        title="Transaction System Overview"
        subtitle="Global transaction monitoring and financial network activity indicators"
      />

      {isLoading && !metrics ? (
        <LoadingState message="Aggregating financial system metrics..." />
      ) : error ? (
        <ErrorState message={error} onRetry={() => fetchMetrics(timeRange)} />
      ) : metrics ? (
        <>
          {/* KPI Row: Maximum 4 primary metrics */}
          <div className="metrics-grid">
            <MetricCard
              label="Total Transaction Value"
              value={formatCurrency(metrics.totalTransactionValue)}
              subtext={`Throughput across ${timeRange} window`}
            />
            <MetricCard
              label="Total Transactions"
              value={metrics.totalTransactions.toLocaleString()}
              subtext="Aggregated ledger flows"
            />
            <MetricCard
              label="Active Accounts"
              value={metrics.activeAccounts.toLocaleString()}
              subtext="Monitored banking nodes"
            />
            <MetricCard
              label="Suspicious Accounts"
              value={metrics.suspiciousAccounts.toLocaleString()}
              subtext="Exceeding 0.3781 risk threshold"
              highlight={true}
            />
          </div>

          {/* Charts Row: Activity Chart + Distribution Chart */}
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: '1.6fr 1fr',
              gap: 16,
              marginBottom: 20,
            }}
          >
            <ActivityChart
              data={metrics.activityOverTime}
              timeRange={timeRange}
              onTimeRangeChange={setTimeRange}
            />
            <DistributionChart distribution={metrics.categoryDistribution} />
          </div>

          {/* Top Accounts Compact Table */}
          <TopAccountsTable accounts={metrics.topAccounts} />
        </>
      ) : null}
    </div>
  );
};
