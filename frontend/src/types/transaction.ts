export type TransactionDirection = 'INCOMING' | 'OUTGOING' | 'INTERNAL';

export interface Transaction {
  id: string;
  timestamp: string;
  sourceAccount: string;
  sourceAccountId: string;
  targetAccount: string;
  targetAccountId: string;
  amount: number;
  direction: TransactionDirection;
  riskScore?: number;
  currency: string;
  paymentFormat?: string;
  isFlagged?: boolean;
}
