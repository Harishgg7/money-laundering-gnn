import { DashboardMetrics, TimeRangeFilter } from '../types';
import { mockDashboardData } from '../data/mock/dashboardData';
import { request } from './apiClient';

export const dashboardService = {
  async getMetrics(range: TimeRangeFilter = '30D'): Promise<DashboardMetrics> {
    return request<DashboardMetrics>(
      `/api/dashboard?range=${range}`,
      undefined,
      () => mockDashboardData[range] || mockDashboardData['30D']
    );
  },
};
