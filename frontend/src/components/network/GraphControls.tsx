import React, { useState, useEffect, useRef } from 'react';
import { RiskLevel } from '../../types';
import { Search, RotateCcw, X, ArrowRight } from 'lucide-react';
import { networkService } from '../../services/networkService';

interface SearchResult {
  accountId: string;
  nodeId: number;
  riskScore: number;
  riskLevel: RiskLevel;
  title?: string;
}

interface GraphControlsProps {
  searchQuery: string;
  onSearchChange: (q: string) => void;
  onSelectAccount?: (accountId: string) => void;
  riskFilter: 'ALL' | RiskLevel;
  onRiskFilterChange: (risk: 'ALL' | RiskLevel) => void;
  onReset: () => void;
}

export const GraphControls: React.FC<GraphControlsProps> = ({
  searchQuery,
  onSearchChange,
  onSelectAccount,
  riskFilter,
  onRiskFilterChange,
  onReset,
}) => {
  const [suggestions, setSuggestions] = useState<SearchResult[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const dropdownRef = useRef<HTMLDivElement | null>(null);
  const suggestionsQueryRef = useRef<string>('');
  const justSelectedRef = useRef<boolean>(false);

  // Debounced autocomplete search
  useEffect(() => {
    let isCancelled = false;
    const q = searchQuery.trim();
    if (!q || q.length < 2) {
      setSuggestions([]);
      setIsOpen(false);
      suggestionsQueryRef.current = '';
      return;
    }

    if (justSelectedRef.current) {
      justSelectedRef.current = false;
      setIsSearching(false);
      return;
    }

    const timer = setTimeout(async () => {
      setIsSearching(true);
      try {
        const results = await networkService.searchAccounts(q, 6);
        if (!isCancelled && searchQuery.trim() === q) {
          setSuggestions(results);
          suggestionsQueryRef.current = q;
          setIsOpen(true);
        }
      } catch {
        if (!isCancelled) {
          setSuggestions([]);
        }
      } finally {
        if (!isCancelled) {
          setIsSearching(false);
        }
      }
    }, 220);

    return () => {
      isCancelled = true;
      clearTimeout(timer);
    };
  }, [searchQuery]);

  // Close dropdown on outside click
  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleOutsideClick);
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, []);

  const handleSelect = (accId: string) => {
    if (accId !== searchQuery) {
      justSelectedRef.current = true;
      onSearchChange(accId);
    }
    setIsOpen(false);
    if (onSelectAccount) {
      onSelectAccount(accId);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      const trimmed = searchQuery.trim();
      const exactMatch = suggestions.find(
        (s) => s.accountId.toLowerCase() === trimmed.toLowerCase()
      );
      if (exactMatch) {
        handleSelect(exactMatch.accountId);
      } else if (
        suggestions.length > 0 &&
        suggestionsQueryRef.current.toLowerCase() === trimmed.toLowerCase()
      ) {
        handleSelect(suggestions[0].accountId);
      } else if (trimmed) {
        handleSelect(trimmed);
      }
    } else if (e.key === 'Escape') {
      setIsOpen(false);
    }
  };

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: 12,
        padding: '10px 14px',
        backgroundColor: 'var(--bg-surface)',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-sm)',
        marginBottom: 14,
        boxShadow: 'var(--shadow-xs)',
      }}
    >
      {/* Search Input with Autocomplete Dropdown */}
      <div ref={dropdownRef} style={{ position: 'relative', flex: 1, maxWidth: 360 }}>
        <div style={{ position: 'relative', width: '100%' }}>
          <Search
            size={13}
            color="var(--text-muted)"
            style={{ position: 'absolute', left: 9, top: '50%', transform: 'translateY(-50%)' }}
          />
          <input
            type="text"
            className="input-control"
            placeholder="Search account (e.g. 8054A41C0, 8000323C0)..."
            value={searchQuery}
            onChange={(e) => {
              justSelectedRef.current = false;
              onSearchChange(e.target.value);
            }}
            onKeyDown={handleKeyDown}
            onFocus={() => {
              if (suggestions.length > 0) setIsOpen(true);
            }}
            style={{ width: '100%', paddingLeft: 28, paddingRight: searchQuery ? 28 : 8, fontSize: 11.5 }}
          />
          {searchQuery && (
            <button
              onClick={() => {
                onSearchChange('');
                setSuggestions([]);
                setIsOpen(false);
              }}
              style={{
                position: 'absolute',
                right: 7,
                top: '50%',
                transform: 'translateY(-50%)',
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                padding: 2,
                color: 'var(--text-muted)',
              }}
              title="Clear search"
            >
              <X size={12} />
            </button>
          )}
        </div>

        {/* Suggestion Dropdown */}
        {isOpen && (
          <div
            style={{
              position: 'absolute',
              top: 'calc(100% + 4px)',
              left: 0,
              right: 0,
              backgroundColor: '#ffffff',
              border: '1px solid var(--border-default)',
              borderRadius: 'var(--radius-sm)',
              boxShadow: '0 4px 14px rgba(0, 0, 0, 0.10)',
              zIndex: 35,
              maxHeight: 250,
              overflowY: 'auto',
            }}
          >
            {isSearching ? (
              <div style={{ padding: '8px 12px', fontSize: 11, color: 'var(--text-muted)' }}>
                Searching 515K network...
              </div>
            ) : suggestions.length > 0 ? (
              <div>
                <div style={{ padding: '6px 10px', fontSize: 10, color: 'var(--text-muted)', borderBottom: '1px solid var(--border-subtle)', textTransform: 'uppercase' }}>
                  Matching Accounts
                </div>
                {suggestions.map((item) => (
                  <div
                    key={item.accountId}
                    onClick={() => handleSelect(item.accountId)}
                    style={{
                      padding: '8px 12px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      cursor: 'pointer',
                      borderBottom: '1px solid var(--border-subtle)',
                      transition: 'background-color 0.15s ease',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'var(--bg-surface-hover)')}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'transparent')}
                  >
                    <div style={{ display: 'flex', flexDirection: 'column' }}>
                      <span className="mono" style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)' }}>
                        {item.accountId}
                      </span>
                      <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>
                        Node #{item.nodeId}
                      </span>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span
                        style={{
                          fontSize: 9.5,
                          fontWeight: 600,
                          padding: '1px 5px',
                          borderRadius: 3,
                          backgroundColor:
                            item.riskLevel === 'HIGH' ? '#fee2e2' : item.riskLevel === 'MEDIUM' ? '#fef3c7' : '#dcfce7',
                          color:
                            item.riskLevel === 'HIGH' ? '#991b1b' : item.riskLevel === 'MEDIUM' ? '#92400e' : '#166534',
                        }}
                      >
                        {item.riskLevel} ({(item.riskScore * 100).toFixed(0)}%)
                      </span>
                      <ArrowRight size={11} color="var(--text-muted)" />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ padding: '8px 12px', fontSize: 11, color: 'var(--text-muted)' }}>
                No accounts match &ldquo;{searchQuery}&rdquo;
              </div>
            )}
          </div>
        )}
      </div>

      {/* Filter and Reset Actions */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
          <span style={{ color: 'var(--text-muted)' }}>Risk:</span>
          <select
            className="input-control"
            value={riskFilter}
            onChange={(e) => onRiskFilterChange(e.target.value as 'ALL' | RiskLevel)}
            style={{ fontSize: 11.5, padding: '4px 6px' }}
          >
            <option value="ALL">All Accounts</option>
            <option value="HIGH">High Risk (&ge;0.38)</option>
            <option value="MEDIUM">Medium Risk</option>
            <option value="LOW">Low Risk</option>
          </select>
        </div>

        <button
          className="btn btn-secondary btn-sm"
          onClick={onReset}
          title="Reset graph filters and recenter"
        >
          <RotateCcw size={12} />
          <span>Reset View</span>
        </button>
      </div>
    </div>
  );
};
