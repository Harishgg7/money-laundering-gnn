import { Investigation, InvestigationStatus } from '../types';
import { mockInvestigations } from '../data/mock/investigationsData';
import { request } from './apiClient';

// In-memory store for mutable status changes during session
let investigationsStore = [...mockInvestigations];

export const investigationService = {
  async getInvestigations(): Promise<Investigation[]> {
    return request<Investigation[]>(
      '/api/investigations',
      undefined,
      () => [...investigationsStore]
    );
  },

  async getInvestigationById(idOrCaseId: string): Promise<Investigation | null> {
    return request<Investigation | null>(
      `/api/investigations/${encodeURIComponent(idOrCaseId)}`,
      undefined,
      () => {
        const found = investigationsStore.find(
          (inv) =>
            inv.id === idOrCaseId ||
            inv.caseId === idOrCaseId ||
            inv.accountId === idOrCaseId ||
            inv.nodeId.toString() === idOrCaseId
        );
        return found ? { ...found } : null;
      }
    );
  },

  async updateStatus(id: string, newStatus: InvestigationStatus): Promise<boolean> {
    return request<{ success: boolean }>(
      `/api/investigations/${encodeURIComponent(id)}/status`,
      {
        method: 'POST',
        body: JSON.stringify({ status: newStatus }),
      },
      () => {
        const index = investigationsStore.findIndex(
          (inv) => inv.id === id || inv.caseId === id || inv.accountId === id
        );
        if (index >= 0) {
          investigationsStore[index] = {
            ...investigationsStore[index],
            status: newStatus,
            updatedAt: new Date().toISOString(),
          };
          return { success: true };
        }
        return { success: false };
      }
    ).then((res) => res.success);
  },
};
