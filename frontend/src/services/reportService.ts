import { Report, CreateReportInput } from '../types';
import { mockReports } from '../data/mock/reportsData';
import { request } from './apiClient';

let reportsStore = [...mockReports];

export const reportService = {
  async getReports(): Promise<Report[]> {
    return request<Report[]>(
      '/api/reports',
      undefined,
      () => [...reportsStore]
    );
  },

  async getReportById(idOrCaseId: string): Promise<Report | null> {
    return request<Report | null>(
      `/api/reports/${encodeURIComponent(idOrCaseId)}`,
      undefined,
      () => {
        const found = reportsStore.find(
          (r) => r.id === idOrCaseId || r.caseId === idOrCaseId
        );
        return found ? { ...found } : null;
      }
    );
  },

  async saveReport(input: CreateReportInput): Promise<Report> {
    return request<Report>(
      '/api/reports',
      {
        method: 'POST',
        body: JSON.stringify(input),
      },
      () => {
        const newReportId = input.id || `REP-2026-00${reportsStore.length + 42}`;
        const newReport: Report = {
          id: newReportId,
          caseId: input.caseId,
          accountId: input.accountId,
          riskLevel: input.riskLevel,
          decision: input.decision,
          status: 'Filed',
          filedDate: new Date().toISOString().split('T')[0],
          investigatorName: input.investigatorName || 'Harish GG (Lead AML Investigator)',
          summary: input.summary,
          observedPattern: input.observedPattern,
          evidence: input.evidence,
          modelAssessment: input.modelAssessment,
          notes: input.notes,
          createdAt: new Date().toISOString(),
        };
        const existingIdx = reportsStore.findIndex((r) => r.id === newReportId);
        if (existingIdx >= 0) {
          reportsStore[existingIdx] = newReport;
        } else {
          reportsStore = [newReport, ...reportsStore];
        }
        return newReport;
      }
    );
  },

  async updateReport(id: string, updates: Partial<Report>): Promise<Report> {
    return request<Report>(
      `/api/reports/${encodeURIComponent(id)}`,
      {
        method: 'PUT',
        body: JSON.stringify(updates),
      },
      () => {
        const idx = reportsStore.findIndex((r) => r.id === id);
        if (idx >= 0) {
          reportsStore[idx] = { ...reportsStore[idx], ...updates, updatedAt: new Date().toISOString() };
          return reportsStore[idx];
        }
        throw new Error(`Report ${id} not found.`);
      }
    );
  },

  async downloadPdf(reportId: string): Promise<{ success: boolean; filename: string }> {
    const res = await fetch(`/api/reports/${encodeURIComponent(reportId)}/pdf`);
    if (!res.ok) {
      let errMsg = `Server error ${res.status}`;
      try {
        const errJson = await res.json();
        if (errJson && errJson.error) errMsg = errJson.error;
      } catch {
        // ignore
      }
      throw new Error(`PDF generation failed: ${errMsg}`);
    }

    const blob = await res.blob();
    const cleanId = reportId.replace(/\s+/g, '_');
    const filename = `TRACE_REPORT_${cleanId}.pdf`;
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    return { success: true, filename };
  },

  async sendEmail(
    reportId: string,
    recipientEmail: string
  ): Promise<{ success: boolean; status: string; message: string }> {
    const res = await fetch(`/api/reports/${encodeURIComponent(reportId)}/email`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ recipient: recipientEmail }),
    });

    const data = await res.json().catch(() => ({}));
    if (!res.ok && res.status !== 200) {
      return {
        success: false,
        status: data.status || 'failed',
        message: data.message || data.error || `Email delivery failed (${res.status}).`,
      };
    }

    return {
      success: !!data.success,
      status: data.status || (data.success ? 'sent' : 'failed'),
      message: data.message || (data.success ? 'Email dispatched successfully.' : 'Email delivery failed.'),
    };
  },
};
