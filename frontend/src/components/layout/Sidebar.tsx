import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Network, ShieldAlert, FileText, Settings, User } from 'lucide-react';

export const Sidebar: React.FC = () => {
  return (
    <aside className="app-sidebar">
      {/* Brand Header */}
      <div className="sidebar-header">
        <div className="brand-badge">T</div>
        <div>
          <div className="brand-title">TRACE</div>
          <div className="brand-subtitle">AML Investigation & Intelligence</div>
        </div>
      </div>

      {/* Main Four Sections */}
      <nav className="sidebar-nav">
        <NavLink
          to="/dashboard"
          className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
        >
          <div className="nav-link-left">
            <LayoutDashboard size={16} />
            <span>Dashboard</span>
          </div>
        </NavLink>

        <NavLink
          to="/network"
          className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
        >
          <div className="nav-link-left">
            <Network size={16} />
            <span>Network</span>
          </div>
        </NavLink>

        <NavLink
          to="/investigations"
          className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
        >
          <div className="nav-link-left">
            <ShieldAlert size={16} />
            <span>Investigations</span>
          </div>
        </NavLink>

        <NavLink
          to="/reports"
          className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
        >
          <div className="nav-link-left">
            <FileText size={16} />
            <span>Reports</span>
          </div>
        </NavLink>
      </nav>

      {/* Bottom Profile & Settings */}
      <div className="sidebar-footer">
        <div className="user-snippet">
          <div className="user-avatar">HG</div>
          <div className="user-info">
            <div className="user-name">Harish GG</div>
            <div className="user-role">Compliance Tier 2</div>
          </div>
        </div>
        <button
          className="btn btn-secondary btn-sm"
          style={{ padding: '4px 6px', border: 'none', background: 'transparent' }}
          title="Settings"
          onClick={() => alert('Settings: TRACE AML Engine v1.0. Connected to IBM HI-Small benchmark model.')}
        >
          <Settings size={15} color="var(--text-muted)" />
        </button>
      </div>
    </aside>
  );
};
