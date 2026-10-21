import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search } from 'lucide-react';

export const Topbar: React.FC = () => {
  const [query, setQuery] = useState('');
  const navigate = useNavigate();

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    // Direct search to Network graph view with query
    navigate(`/network?search=${encodeURIComponent(query.trim())}`);
  };

  return (
    <header className="app-topbar">
      <form onSubmit={handleSearchSubmit} className="topbar-search">
        <Search size={14} color="var(--text-muted)" />
        <input
          type="text"
          placeholder="Search Account ID (e.g. 8054A41C0) or node..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </form>

      <div className="topbar-actions">
        <div className="status-indicator" title="Analysis services loaded and ready">
          <span className="status-dot-ready" />
          <span>Analysis Service &bull; Ready</span>
        </div>
      </div>
    </header>
  );
};
