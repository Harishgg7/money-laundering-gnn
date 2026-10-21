import React from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Topbar } from './Topbar';

export const AppLayout: React.FC = () => {
  return (
    <div className="app-container">
      <Sidebar />
      <div className="app-main">
        <Topbar />
        <main className="workspace-scroll">
          <Outlet />
        </main>
      </div>
    </div>
  );
};
