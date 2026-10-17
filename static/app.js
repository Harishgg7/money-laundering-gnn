/**
 * TRACE — AML Investigation & Financial Crime Intelligence Platform
 * Client Application Logic connecting directly to FastAPI Backend REST Endpoints.
 * No hardcoded enterprise dataset: All data is fetched dynamically from the GNN & IBM AML dataset.
 */

// Global State
const appState = {
  activeView: 'overview',
  activeCaseId: 'AML-2026-00421',
  activeCaseKey: 'case1',
  activeCaseData: null,
  activeAccountData: null,
  activeTransactions: [],
  activeNetwork: { nodes: [], edges: [] },
  activeExplanation: null,
  activeEvidence: [],
  selectedTxns: new Set(),
  overviewData: null,
  alertsData: [],
  auditLogData: [],
  benchmarksData: null,
  demoCases: {},
  presentationMode: false,
};

// Canvas Graph State
const graphState = {
  canvas: null,
  ctx: null,
  scale: 1.0,
  panX: 0,
  panY: 0,
  isDragging: false,
  dragStart: { x: 0, y: 0 },
  highlightedPath: null,
  hoveredNode: null,
  nodePositions: new Map(),
};

// -----------------------------------------------------------------------------
// Initialization & Event Wiring
// -----------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', async () => {
  initCanvas();
  initNavigation();
  initModals();
  initKeyboardShortcuts();

  // Load initial backend datasets
  await loadInitialData();
});

async function loadInitialData() {
  try {
    // 1. Fetch Overview & Metrics
    await refreshOverview();

    // 2. Fetch Curated Demo Cases
    const demoResp = await fetch('/api/demo_cases');
    if (demoResp.ok) {
      const demoJson = await demoResp.json();
      appState.demoCases = demoJson.cases || {};
    }

    // 3. Load Active Case (Default Case 1)
    await loadCase('case1');

    // 4. Fetch Full Alerts Queue
    await refreshAlerts();

    // 5. Fetch Audit Trail
    await refreshAuditLog();

    // 6. Fetch Model Benchmarks
    await refreshBenchmarks();

  } catch (err) {
    console.error('[TRACE] Initialization error:', err);
  }
}

// -----------------------------------------------------------------------------
// Navigation & View Switching
// -----------------------------------------------------------------------------

function initNavigation() {
  // Sidebar Links
  document.querySelectorAll('.sidebar-nav .nav-item').forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const viewId = item.dataset.view;
      switchView(viewId);
    });
  });

  // Topbar Demo Case Pills
  document.querySelectorAll('.case-pill-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const caseKey = btn.dataset.case;
      document.querySelectorAll('.case-pill-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      await loadCase(caseKey);
    });
  });

  // Presentation Mode Toggle
  const presBtn = document.getElementById('btnTogglePresentation');
  if (presBtn) {
    presBtn.addEventListener('click', togglePresentationMode);
  }

  // Global Search
  const searchInput = document.getElementById('globalSearchInput');
  if (searchInput) {
    searchInput.addEventListener('keydown', async (e) => {
      if (e.key === 'Enter') {
        const query = searchInput.value.trim();
        if (!query) return;
        await handleGlobalSearch(query);
      }
    });
  }

  // Filters in Overview
  const riskFilter = document.getElementById('filterAlertRisk');
  const statusFilter = document.getElementById('filterAlertStatus');
  if (riskFilter) riskFilter.addEventListener('change', () => filterOverviewAlerts());
  if (statusFilter) statusFilter.addEventListener('change', () => filterOverviewAlerts());

  // Filter in Alerts tab
  const kwFilter = document.getElementById('filterAlertKeyword');
  if (kwFilter) {
    kwFilter.addEventListener('input', () => {
      const q = kwFilter.value.toLowerCase().trim();
      const rows = document.querySelectorAll('#tbodyFullAlerts tr');
      rows.forEach(r => {
        const text = r.innerText.toLowerCase();
        r.style.display = text.includes(q) ? '' : 'none';
      });
    });
  }

  // Trace Funds Action Button
  const btnTrace = document.getElementById('btnExecuteTrace');
  if (btnTrace) {
    btnTrace.addEventListener('click', executeTraceFunds);
  }

  // Save Investigator Note Action Button
  const btnNote = document.getElementById('btnSaveInvestigatorNote');
  if (btnNote) {
    btnNote.addEventListener('click', saveInvestigatorNote);
  }

  // Recommendation Selector in Report Tab
  const selRec = document.getElementById('selReportRecommendation');
  if (selRec) {
    selRec.addEventListener('change', () => {
      updateReportPreview();
    });
  }
}

function switchView(viewId) {
  appState.activeView = viewId;
  document.querySelectorAll('.view-container').forEach(c => c.classList.remove('active'));
  document.querySelectorAll('.sidebar-nav .nav-item').forEach(n => n.classList.remove('active'));

  const targetView = document.getElementById('view' + viewId.charAt(0).toUpperCase() + viewId.slice(1));
  const targetNav = document.getElementById('nav' + viewId.charAt(0).toUpperCase() + viewId.slice(1));

  if (targetView) targetView.classList.add('active');
  if (targetNav) targetNav.classList.add('active');

  // Trigger redraw if entering Network or Case view
  if (viewId === 'case' || viewId === 'network') {
    setTimeout(fitGraphToViewport, 80);
  }
}

function switchCaseTab(tabId) {
  document.querySelectorAll('.case-tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.case-tab-content').forEach(c => c.classList.remove('active'));

  const targetBtn = document.querySelector(`.case-tab-btn[data-casetab="${tabId}"]`);
  const targetTab = document.getElementById('caseTab' + tabId.charAt(0).toUpperCase() + tabId.slice(1));

  if (targetBtn) targetBtn.classList.add('active');
  if (targetTab) targetTab.classList.add('active');

  if (tabId === 'network') {
    setTimeout(fitGraphToViewport, 80);
  }
}

// -----------------------------------------------------------------------------
// Case & Account Investigation Loader
// -----------------------------------------------------------------------------

async function loadCase(caseIdentifier) {
  try {
    // 1. Fetch Case Details from backend
    const resp = await fetch(`/api/cases/${encodeURIComponent(caseIdentifier)}`);
    if (!resp.ok) {
      throw new Error(`Unable to load case ${caseIdentifier}`);
    }
    const caseData = await resp.json();
    appState.activeCaseData = caseData;
    appState.activeCaseId = caseData.case_id;

    // 2. Fetch Subject Account Details
    const acctResp = await fetch(`/api/accounts/${encodeURIComponent(caseData.account_key)}`);
    if (acctResp.ok) {
      appState.activeAccountData = await acctResp.json();
    } else {
      appState.activeAccountData = caseData.subject_account || {};
    }

    // 3. Fetch Transactions
    const txResp = await fetch(`/api/accounts/${encodeURIComponent(caseData.account_key)}/transactions`);
    if (txResp.ok) {
      appState.activeTransactions = await txResp.json();
    } else {
      appState.activeTransactions = [];
    }

    // 4. Fetch Network Topology
    const netResp = await fetch(`/api/accounts/${encodeURIComponent(caseData.account_key)}/network?depth=1`);
    if (netResp.ok) {
      appState.activeNetwork = await netResp.json();
    } else {
      appState.activeNetwork = { nodes: [], edges: [] };
    }

    // 5. Fetch Explanation
    const expResp = await fetch(`/api/explanations/${encodeURIComponent(caseData.account_key)}`);
    if (expResp.ok) {
      appState.activeExplanation = await expResp.json();
    } else {
      appState.activeExplanation = null;
    }

    // 6. Update UI Components
    renderCaseHeader(caseData, appState.activeAccountData);
    renderCaseOverviewTab(caseData, appState.activeAccountData, appState.activeExplanation);
    renderTransactionsTab(appState.activeTransactions);
    renderNetworkTab(appState.activeNetwork, appState.activeAccountData);
    renderEvidenceTab();
    renderReportTab(caseData, appState.activeAccountData, appState.activeExplanation);

    // Update active state in topbar
    if (caseIdentifier === 'case1' || caseIdentifier === '496618') {
      updateActiveCasePill('topBtnCase1');
    } else if (caseIdentifier === 'case2' || caseIdentifier === '6') {
      updateActiveCasePill('topBtnCase2');
    } else if (caseIdentifier === 'case3' || caseIdentifier === '91') {
      updateActiveCasePill('topBtnCase3');
    }

  } catch (err) {
    console.error('[TRACE] Failed to load case:', err);
    alert('Unable to load investigation case details. Please try again.');
  }
}

function updateActiveCasePill(buttonId) {
  document.querySelectorAll('.case-pill-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById(buttonId);
  if (btn) btn.classList.add('active');
}

// -----------------------------------------------------------------------------
// Rendering Case Views
// -----------------------------------------------------------------------------

function renderCaseHeader(c, acct) {
  // Title Card
  const codeEl = document.getElementById('activeCaseCode');
  const riskBadge = document.getElementById('activeCaseRiskBadge');
  const subtitleEl = document.getElementById('activeCaseSubtitle');
  const statusBadge = document.getElementById('activeCaseStatusBadge');

  if (codeEl) codeEl.innerText = c.case_id || 'AML-CASE';
  if (subtitleEl) subtitleEl.innerText = `${acct.customer_name || 'Commercial Entity'} (${acct.typology || c.alert_type})`;

  const riskPct = acct.risk_pct !== undefined ? acct.risk_pct : Math.round((acct.risk_score || 0) * 100);
  const isHigh = (acct.risk_score || 0) >= (acct.operating_threshold || 0.378);

  if (riskBadge) {
    riskBadge.className = `badge ${isHigh ? 'badge-high' : 'badge-low'}`;
    riskBadge.innerText = `${isHigh ? 'HIGH RISK' : 'LOW RISK'} • ${riskPct}%`;
  }

  if (statusBadge) {
    statusBadge.className = `badge-status badge-${(c.status || 'open').toLowerCase().replace(/\s+/g, '')}`;
    statusBadge.innerText = `● ${c.status || 'OPEN'}`;
  }

  // Subject Account Summary Strip
  setText('subAcctKey', acct.account_key || '—');
  setText('subNodeMeta', `${acct.customer_name} • ${acct.bank_name}`);
  setText('subRiskScoreText', `${riskPct}%`);
  
  const tierPill = document.getElementById('subRiskTierPill');
  if (tierPill) {
    tierPill.className = `badge ${isHigh ? 'badge-high' : 'badge-low'}`;
    tierPill.innerText = isHigh ? 'High Conviction' : 'Nominal Risk';
  }

  setText('subCounterparties', `${acct.counterparty_count || (acct.in_degree + acct.out_degree)} accounts`);
  setText('subDegBreakdown', `${acct.in_degree || 0} in • ${acct.out_degree || 0} out direct`);
  setText('subTotalInflow', `$${(acct.total_inflow || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
  setText('subTotalOutflow', `$${(acct.total_outflow || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
}

function renderCaseOverviewTab(c, acct, explanation) {
  // Summary Paragraph with careful compliance language
  const summaryEl = document.getElementById('caseSummaryParagraph');
  if (summaryEl) {
    const isHigh = (acct.risk_score || 0) >= (acct.operating_threshold || 0.378);
    if (isHigh) {
      summaryEl.innerText = (
        `Account ${acct.account_key} (${acct.customer_name}) has been flagged by the automated graph surveillance ` +
        `system because its observed transaction patterns exceeded the calibrated operating threshold (${acct.operating_threshold || 0.378}). ` +
        `The GNN evaluated structural money flows and isolated relational characteristics consistent with a ${acct.typology}. ` +
        `The investigator should examine the counterparty relationships and evidentiary records before determining statutory escalation.`
      );
    } else {
      summaryEl.innerText = (
        `Account ${acct.account_key} (${acct.customer_name}) exhibits routine counterparty activity with an assessed ` +
        `risk conviction of ${(acct.risk_score * 100).toFixed(1)}%, well below the operational decision threshold (${acct.operating_threshold || 0.378}). ` +
        `No suspicious multi-hop structuring or anomalous holding balances were detected.`
      );
    }
  }

  // Why was this alert generated reasons list
  const reasonList = document.getElementById('caseReasonList');
  if (reasonList) {
    reasonList.innerHTML = '';
    const reasons = [];

    if ((acct.risk_score || 0) >= (acct.operating_threshold || 0.378)) {
      reasons.push({
        title: 'Model Operating Threshold Exceeded',
        desc: `Assessed risk score (${(acct.risk_score * 100).toFixed(1)}%) surpasses validation-tuned threshold (${acct.operating_threshold || 0.378}) for financial crime review.`
      });
      reasons.push({
        title: 'Relational Graph Pattern Identified',
        desc: `GraphSAGE 3-hop inductive message passing extracted structural pattern: ${acct.typology}.`
      });
      reasons.push({
        title: 'Counterparty Network Concentration',
        desc: `Observed asymmetric pass-through velocity across ${acct.counterparty_count || (acct.in_degree + acct.out_degree)} connected accounts.`
      });
    } else {
      reasons.push({
        title: 'Routine Commercial Flow',
        desc: 'Transactions align with expected ordinary commercial volume and counterparty diversity.'
      });
      reasons.push({
        title: 'Low Model Conviction',
        desc: `Risk score (${(acct.risk_score * 100).toFixed(1)}%) remains firmly beneath trigger threshold.`
      });
    }

    reasons.forEach(r => {
      const li = document.createElement('li');
      li.className = 'reason-item';
      li.innerHTML = `
        <div class="reason-dot"></div>
        <div>
          <b>${escapeHtml(r.title)}</b>
          ${escapeHtml(r.desc)}
        </div>
      `;
      reasonList.appendChild(li);
    });
  }

  // Timeline
  const timelineEl = document.getElementById('caseTimelineList');
  if (timelineEl) {
    timelineEl.innerHTML = '';
    const timelineItems = (c.curated_data && c.curated_data.timeline) ? c.curated_data.timeline : [];
    
    if (timelineItems.length === 0 && appState.activeTransactions.length > 0) {
      // Build from transactions
      appState.activeTransactions.slice(0, 6).forEach(tx => {
        timelineItems.push({
          time: tx.date,
          title: `${tx.amount} ${tx.type} (${tx.direction})`,
          desc: `${tx.from} → ${tx.to}`
        });
      });
    }

    if (timelineItems.length === 0) {
      timelineEl.innerHTML = '<div style="color:var(--text-muted); font-size:12px;">No recent transaction timeline events recorded.</div>';
    } else {
      timelineItems.forEach(t => {
        const item = document.createElement('div');
        item.style.cssText = 'display:flex; gap:12px; font-size:12.5px;';
        item.innerHTML = `
          <div style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted); width:110px; flex-shrink:0;">${escapeHtml(t.time)}</div>
          <div style="border-left:2px solid var(--border-subtle); padding-left:10px;">
            <div style="font-weight:600; color:var(--text-primary);">${escapeHtml(t.title)}</div>
            <div style="color:var(--text-secondary); font-size:11.5px;">${escapeHtml(t.desc)}</div>
          </div>
        `;
        timelineEl.appendChild(item);
      });
    }
  }
}

function renderTransactionsTab(txs) {
  const badge = document.getElementById('txnCountBadge');
  if (badge) badge.innerText = txs.length;

  const tbody = document.getElementById('tbodyCaseTransactions');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (txs.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; padding:20px; color:var(--text-muted);">No transaction relationships found for this account.</td></tr>';
    return;
  }

  txs.forEach(t => {
    const tr = document.createElement('tr');
    tr.id = `rowTxn_${t.id}`;
    
    const isSelected = appState.selectedTxns.has(t.id);
    if (isSelected) tr.classList.add('selected');

    const riskBadgeClass = t.risk === 'HIGH' ? 'badge-high' : (t.risk === 'MED' ? 'badge-med' : 'badge-low');
    const typeLabel = t.is_aggregated_relationship ? 'Aggregated Relationship' : t.type;

    tr.innerHTML = `
      <td><input type="checkbox" class="chk-txn" data-txid="${t.id}" ${isSelected ? 'checked' : ''} onclick="event.stopPropagation(); toggleTxnSelection('${t.id}')"></td>
      <td class="mono" style="font-weight:600; color:var(--primary-blue); cursor:pointer;" onclick="openSidePanel('${t.id}')">${escapeHtml(t.id)}</td>
      <td style="font-size:12px;">${escapeHtml(t.date)}</td>
      <td class="mono" title="${escapeHtml(t.from_customer || '')}">${escapeHtml(t.from)}</td>
      <td><span class="badge ${t.direction === 'INCOMING' ? 'badge-low' : 'badge-med'}">${escapeHtml(t.direction)}</span></td>
      <td class="mono" title="${escapeHtml(t.to_customer || '')}">${escapeHtml(t.to)}</td>
      <td style="font-weight:600;">${escapeHtml(t.amount)}</td>
      <td style="font-size:11px; color:var(--text-secondary);">${escapeHtml(typeLabel)}</td>
      <td><span class="badge ${riskBadgeClass}">${escapeHtml(t.risk)}</span></td>
    `;
    tr.addEventListener('click', () => openSidePanel(t.id));
    tbody.appendChild(tr);
  });
}

function toggleTxnSelection(txId) {
  if (appState.selectedTxns.has(txId)) {
    appState.selectedTxns.delete(txId);
  } else {
    appState.selectedTxns.add(txId);
  }

  const row = document.getElementById(`rowTxn_${txId}`);
  if (row) row.classList.toggle('selected', appState.selectedTxns.has(txId));

  updateStickyBar();
}

function updateStickyBar() {
  const bar = document.getElementById('stickyEvidenceBar');
  const countLbl = document.getElementById('lblSelectedTxnCount');
  if (!bar || !countLbl) return;

  countLbl.innerText = appState.selectedTxns.size;
  if (appState.selectedTxns.size > 0) {
    bar.classList.add('show');
  } else {
    bar.classList.remove('show');
  }
}

function clearSelectedTxns() {
  appState.selectedTxns.clear();
  document.querySelectorAll('.chk-txn').forEach(c => c.checked = false);
  document.querySelectorAll('#tbodyCaseTransactions tr').forEach(r => r.classList.remove('selected'));
  updateStickyBar();
}

// -----------------------------------------------------------------------------
// Network Canvas Visualizer
// -----------------------------------------------------------------------------

function initCanvas() {
  const c = document.getElementById('investigationCanvas');
  if (!c) return;
  graphState.canvas = c;
  graphState.ctx = c.getContext('2d');

  function resize() {
    const rect = c.parentElement.getBoundingClientRect();
    c.width = rect.width * window.devicePixelRatio;
    c.height = rect.height * window.devicePixelRatio;
    drawNetworkGraph();
  }

  window.addEventListener('resize', resize);
  setTimeout(resize, 50);

  // Pan and drag
  c.addEventListener('mousedown', (e) => {
    graphState.isDragging = true;
    graphState.dragStart = { x: e.clientX - graphState.panX, y: e.clientY - graphState.panY };
  });

  window.addEventListener('mousemove', (e) => {
    if (graphState.isDragging) {
      graphState.panX = e.clientX - graphState.dragStart.x;
      graphState.panY = e.clientY - graphState.dragStart.y;
      drawNetworkGraph();
    }
  });

  window.addEventListener('mouseup', () => {
    graphState.isDragging = false;
  });

  // Scroll to zoom
  c.addEventListener('wheel', (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.1 : 0.9;
    graphState.scale = Math.max(0.3, Math.min(3.5, graphState.scale * factor));
    drawNetworkGraph();
  });

  // Floating Controls
  const btnZoomIn = document.getElementById('btnZoomIn');
  const btnZoomOut = document.getElementById('btnZoomOut');
  const btnFit = document.getElementById('btnFitGraph');

  if (btnZoomIn) btnZoomIn.addEventListener('click', () => { graphState.scale *= 1.2; drawNetworkGraph(); });
  if (btnZoomOut) btnZoomOut.addEventListener('click', () => { graphState.scale /= 1.2; drawNetworkGraph(); });
  if (btnFit) btnFit.addEventListener('click', fitGraphToViewport);
}

function renderNetworkTab(net, subjectAcct) {
  // Populate dropdowns for Trace Money Flow
  const selSrc = document.getElementById('selTraceSource');
  const selDst = document.getElementById('selTraceTarget');

  if (selSrc && selDst) {
    selSrc.innerHTML = '';
    selDst.innerHTML = '';

    const nodes = net.nodes || [];
    nodes.forEach(n => {
      const opt1 = document.createElement('option');
      opt1.value = n.key || n.id;
      opt1.text = `${n.key} (${n.customer_name || n.role || 'Account'})`;
      selSrc.appendChild(opt1);

      const opt2 = document.createElement('option');
      opt2.value = n.key || n.id;
      opt2.text = `${n.key} (${n.customer_name || n.role || 'Account'})`;
      if (n.is_target) opt2.selected = true;
      selDst.appendChild(opt2);
    });
  }

  // Mini Explainer table on the side
  const miniTable = document.getElementById('miniExplainerTable');
  if (miniTable) {
    miniTable.innerHTML = '';
    const edges = net.edges || [];
    const decisiveEdges = edges.filter(e => e.importance >= 0.4 || e.is_decisive).slice(0, 4);

    if (decisiveEdges.length === 0) {
      miniTable.innerHTML = '<div style="color:var(--text-muted); font-size:11px;">No high-weight edge attributions.</div>';
    } else {
      decisiveEdges.forEach(e => {
        const item = document.createElement('div');
        item.style.cssText = 'display:flex; justify-content:space-between; padding:3px 0; border-bottom:1px dashed var(--border-subtle);';
        item.innerHTML = `
          <span>${escapeHtml(e.source_key || e.source)} &rarr; ${escapeHtml(e.target_key || e.target)}</span>
          <b style="color:var(--primary-blue);">${(e.importance * 100).toFixed(1)}%</b>
        `;
        miniTable.appendChild(item);
      });
    }
  }

  // Position nodes and render
  layoutNetworkNodes(net, subjectAcct);
  fitGraphToViewport();
}

function layoutNetworkNodes(net, subjectAcct) {
  graphState.nodePositions.clear();
  const nodes = net.nodes || [];
  const count = nodes.length;

  if (count === 0) return;

  const targetNode = nodes.find(n => n.is_target) || nodes[0];
  const centerPos = { x: 400, y: 220 };
  graphState.nodePositions.set(targetNode.id, centerPos);

  // Position other nodes radially or hierarchically
  const otherNodes = nodes.filter(n => n.id !== targetNode.id);
  const radius = 180;
  otherNodes.forEach((node, i) => {
    // If predefined x, y in curated cases, use them
    if (node.x && node.y) {
      graphState.nodePositions.set(node.id, { x: node.x, y: node.y });
    } else {
      const angle = (2 * Math.PI * i) / otherNodes.length;
      graphState.nodePositions.set(node.id, {
        x: centerPos.x + radius * Math.cos(angle),
        y: centerPos.y + radius * Math.sin(angle)
      });
    }
  });
}

function fitGraphToViewport() {
  if (!graphState.canvas) return;
  graphState.scale = 1.0;
  graphState.panX = 0;
  graphState.panY = 0;
  drawNetworkGraph();
}

function drawNetworkGraph() {
  const c = graphState.canvas;
  const ctx = graphState.ctx;
  if (!c || !ctx) return;

  const dpr = window.devicePixelRatio || 1;
  ctx.save();
  ctx.clearRect(0, 0, c.width, c.height);

  ctx.scale(dpr * graphState.scale, dpr * graphState.scale);
  ctx.translate(graphState.panX / graphState.scale, graphState.panY / graphState.scale);

  const net = appState.activeNetwork;
  const nodes = net.nodes || [];
  const edges = net.edges || [];

  // Draw Edges
  edges.forEach(e => {
    const sPos = graphState.nodePositions.get(e.source);
    const dPos = graphState.nodePositions.get(e.target);
    if (!sPos || !dPos) return;

    const isHighPath = graphState.highlightedPath && graphState.highlightedPath.has(`${e.source}-${e.target}`);
    const isDecisive = e.importance >= 0.5 || e.is_decisive;

    ctx.beginPath();
    ctx.moveTo(sPos.x, sPos.y);
    ctx.lineTo(dPos.x, dPos.y);

    if (isHighPath) {
      ctx.strokeStyle = '#2563EB';
      ctx.lineWidth = 3.5;
    } else if (isDecisive) {
      ctx.strokeStyle = '#D92D20';
      ctx.lineWidth = 2.2;
    } else {
      ctx.strokeStyle = '#D1D5DB';
      ctx.lineWidth = 1.2;
    }
    ctx.stroke();

    // Draw Arrowhead
    const angle = Math.atan2(dPos.y - sPos.y, dPos.x - sPos.x);
    const arrowDist = 18;
    const arrowX = dPos.x - arrowDist * Math.cos(angle);
    const arrowY = dPos.y - arrowDist * Math.sin(angle);

    ctx.beginPath();
    ctx.moveTo(arrowX, arrowY);
    ctx.lineTo(arrowX - 8 * Math.cos(angle - Math.PI / 6), arrowY - 8 * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(arrowX - 8 * Math.cos(angle + Math.PI / 6), arrowY - 8 * Math.sin(angle + Math.PI / 6));
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fill();
  });

  // Draw Nodes
  nodes.forEach(n => {
    const pos = graphState.nodePositions.get(n.id);
    if (!pos) return;

    ctx.beginPath();
    const radius = n.is_target ? 16 : 11;
    ctx.arc(pos.x, pos.y, radius, 0, 2 * Math.PI);

    if (n.is_target) {
      ctx.fillStyle = '#2457C5';
      ctx.strokeStyle = '#1D46A3';
    } else if (n.risk_cat === 'HIGH' || n.risk_score >= 0.378) {
      ctx.fillStyle = '#D92D20';
      ctx.strokeStyle = '#B42318';
    } else if (n.risk_cat === 'MED') {
      ctx.fillStyle = '#DC9A00';
      ctx.strokeStyle = '#B54708';
    } else {
      ctx.fillStyle = '#159455';
      ctx.strokeStyle = '#027A48';
    }

    ctx.lineWidth = 2;
    ctx.fill();
    ctx.stroke();

    // Label
    ctx.fillStyle = '#17202A';
    ctx.font = '500 11px Inter, sans-serif';
    ctx.textAlign = 'center';
    const labelKey = n.key ? n.key.slice(-8) : `ID ${n.id}`;
    ctx.fillText(labelKey, pos.x, pos.y + radius + 13);
  });

  ctx.restore();
}

// -----------------------------------------------------------------------------
// Trace Funds Action
// -----------------------------------------------------------------------------

async function executeTraceFunds() {
  const selSrc = document.getElementById('selTraceSource');
  const selDst = document.getElementById('selTraceTarget');
  const resBox = document.getElementById('traceResultContainer');
  const summaryEl = document.getElementById('traceResultSummary');
  const detailsEl = document.getElementById('tracePathDetails');
  const metricsEl = document.getElementById('tracePathMetrics');

  if (!selSrc || !selDst || !resBox) return;

  const srcVal = selSrc.value;
  const dstVal = selDst.value;

  try {
    const resp = await fetch(`/api/network/${encodeURIComponent(srcVal)}/trace?target=${encodeURIComponent(dstVal)}`);
    if (!resp.ok) {
      throw new Error('Trace API query error');
    }

    const data = await resp.json();
    resBox.style.display = 'block';

    if (data.found && data.path.length > 0) {
      summaryEl.innerText = `${data.hop_count}-hop money flow path confirmed`;
      summaryEl.style.color = 'var(--text-primary)';

      detailsEl.innerHTML = '';
      data.path.forEach((nodeKey, idx) => {
        const step = document.createElement('div');
        step.className = 'path-step';
        step.innerHTML = `
          <span style="color:var(--text-muted); font-size:10px;">HOP 0${idx}</span>
          <span class="mono" style="font-weight:600; font-size:11px;">${escapeHtml(nodeKey)}</span>
        `;
        detailsEl.appendChild(step);
      });

      metricsEl.innerText = `Total Observed Flow: $${data.total_observed_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} • ${data.hop_count} Conduits`;

      // Highlight on canvas
      graphState.highlightedPath = new Set();
      for (let i = 0; i < data.path_nodes.length - 1; i++) {
        graphState.highlightedPath.add(`${data.path_nodes[i]}-${data.path_nodes[i+1]}`);
      }
      drawNetworkGraph();

      // Log in audit log
      await logAuditAction('Network Funds Traced', appState.activeCaseId, `Traced money flow from ${srcVal} to ${dstVal} (${data.hop_count} hops).`);

    } else {
      summaryEl.innerText = 'No supported path found within selected network scope.';
      summaryEl.style.color = 'var(--text-secondary)';
      detailsEl.innerHTML = '<div style="font-size:11.5px; color:var(--text-muted); padding:6px 0;">No active directed transaction conduits connect these two entities.</div>';
      metricsEl.innerText = 'Scope: Up to 5 hops across directed transaction ledger.';
      graphState.highlightedPath = null;
      drawNetworkGraph();
    }

  } catch (err) {
    console.error('[TRACE] Failed to trace funds:', err);
    alert('Unable to trace funds between selected accounts.');
  }
}

// -----------------------------------------------------------------------------
// Evidence Board & Sticky Bar
// -----------------------------------------------------------------------------

function renderEvidenceTab() {
  const counter = document.getElementById('evidenceCounterBadge');
  if (counter) counter.innerText = appState.activeEvidence.length;

  const tbody = document.getElementById('tbodyEvidenceTxns');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (appState.activeEvidence.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:var(--text-muted); padding:20px;">No transactions added to evidence yet. Select transactions from the Transactions tab.</td></tr>';
    return;
  }

  appState.activeEvidence.forEach((item, idx) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="mono" style="font-weight:600;">${escapeHtml(item.id)}</td>
      <td style="font-size:12px;">${escapeHtml(item.date || '2026-10-17')}</td>
      <td class="mono">${escapeHtml(item.from || item.source_key || 'Source')} &rarr; ${escapeHtml(item.to || item.target_key || 'Target')}</td>
      <td style="font-weight:600;">${escapeHtml(item.amount || '$0.00')}</td>
      <td><button class="btn btn-secondary btn-sm" onclick="removeEvidenceItem('${item.id}')">Remove</button></td>
    `;
    tbody.appendChild(tr);
  });
}

// Button in sticky bar: Add Selected to Evidence
document.addEventListener('click', async (e) => {
  if (e.target && e.target.id === 'btnAddSelectedToEvidence') {
    const count = appState.selectedTxns.size;
    if (count === 0) return;

    appState.selectedTxns.forEach(txId => {
      const tx = appState.activeTransactions.find(t => t.id === txId);
      if (tx && !appState.activeEvidence.some(ev => ev.id === tx.id)) {
        appState.activeEvidence.push(tx);
      }
    });

    clearSelectedTxns();
    renderEvidenceTab();
    updateReportPreview();

    await logAuditAction('Evidence Added', appState.activeCaseId, `Added ${count} transaction(s) to evidentiary docket.`);
    alert(`Successfully added ${count} transaction(s) to the Case Evidence docket.`);
  }
});

async function removeEvidenceItem(itemId) {
  appState.activeEvidence = appState.activeEvidence.filter(e => e.id !== itemId);
  renderEvidenceTab();
  updateReportPreview();
  await logAuditAction('Evidence Removed', appState.activeCaseId, `Removed item ${itemId} from evidentiary docket.`);
}

// -----------------------------------------------------------------------------
// Investigator Notes
// -----------------------------------------------------------------------------

async function saveInvestigatorNote() {
  const area = document.getElementById('investigatorNotesArea');
  if (!area) return;
  const text = area.value.trim();
  if (!text) {
    alert('Please enter note text before saving.');
    return;
  }

  try {
    const resp = await fetch(`/api/cases/${encodeURIComponent(appState.activeCaseId)}/notes`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, investigator: 'H. Vance (Senior AML Analyst)' })
    });

    if (resp.ok) {
      const stamp = document.getElementById('noteSavedTimestamp');
      if (stamp) stamp.innerText = `Last saved: Just now by H. Vance`;
      await logAuditAction('Investigator Note Added', appState.activeCaseId, `Note: ${text.slice(0, 50)}...`);
      updateReportPreview();
      alert('Investigator note successfully saved to case docket.');
    }
  } catch (err) {
    console.error('[TRACE] Failed to save note:', err);
  }
}

// -----------------------------------------------------------------------------
// Report Generation Tab
// -----------------------------------------------------------------------------

function renderReportTab(c, acct, explanation) {
  updateReportPreview();
}

function updateReportPreview() {
  const c = appState.activeCaseData || {};
  const acct = appState.activeAccountData || {};
  const selRec = document.getElementById('selReportRecommendation');
  const recommendation = selRec ? selRec.value : 'Escalate for Further Review';

  setText('repTrackingCode', `SAR-TRACE-2026-${acct.node_id ? String(acct.node_id).padStart(6, '0') : '00421'}`);
  setText('repCaseId', c.case_id || 'AML-2026-00421');
  setText('repStatus', c.status || 'UNDER REVIEW');
  setText('repInstCode', acct.bank_name || 'BANK-0214');
  setText('repAccountKey', acct.account_key || '0214-80C1C8AC0');
  setText('repNodeId', acct.node_id || '496618');
  setText('repRiskScore', `${((acct.risk_score || 0) * 100).toFixed(1)}% (${(acct.risk_score || 0) >= (acct.operating_threshold || 0.378) ? 'HIGH RISK' : 'LOW RISK'})`);

  const summary = (
    `Subject Account ${acct.account_key} (${acct.customer_name}) was flagged by the automated ` +
    `graph surveillance engine because observed multi-hop transaction structures exceeded the configured ` +
    `operational threshold (${acct.operating_threshold || 0.378}). The pattern aligns with a ${acct.typology || 'Suspicious Multi-Party Funnel'} ` +
    `with rapid pass-through velocity.`
  );
  setText('repSummaryText', summary);

  // Evidentiary Table in Report
  const tbody = document.getElementById('repTxnTbody');
  if (tbody) {
    tbody.innerHTML = '';
    const items = appState.activeEvidence.length > 0 ? appState.activeEvidence : appState.activeTransactions.slice(0, 4);
    if (items.length === 0) {
      tbody.innerHTML = '<tr><td colspan="4" style="color:var(--text-muted); text-align:center;">No evidentiary records attached.</td></tr>';
    } else {
      items.forEach(item => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="mono">${escapeHtml(item.id)}</td>
          <td>${escapeHtml(item.date || '2026-10-17')}</td>
          <td class="mono">${escapeHtml(item.from)} &rarr; ${escapeHtml(item.to)}</td>
          <td style="font-weight:600;">${escapeHtml(item.amount)}</td>
        `;
        tbody.appendChild(tr);
      });
    }
  }

  // Notes in report
  const notesArea = document.getElementById('investigatorNotesArea');
  const currentNote = notesArea ? notesArea.value.trim() : '';
  const noteDisplay = document.getElementById('repNotesDisplay');
  if (noteDisplay) {
    noteDisplay.innerText = currentNote ? `"${currentNote}"` : `"Observed transaction structure shows pass-through funneling across institutional banking lines. Recommendation: ${recommendation}."`;
  }
}

// -----------------------------------------------------------------------------
// Overview View & Priority Alerts Queue
// -----------------------------------------------------------------------------

async function refreshOverview() {
  const resp = await fetch('/api/overview');
  if (!resp.ok) return;
  const ov = await resp.json();
  appState.overviewData = ov;

  const alerts = ov.priority_alerts || [];
  renderOverviewAlertsTable(alerts);
}

function renderOverviewAlertsTable(alerts) {
  const tbody = document.getElementById('tbodyOverviewAlerts');
  if (!tbody) return;
  tbody.innerHTML = '';

  alerts.forEach(a => {
    const tr = document.createElement('tr');
    tr.style.cursor = 'pointer';
    
    tr.innerHTML = `
      <td class="mono" style="font-weight:600; color:var(--primary-blue);">${escapeHtml(a.alert_id)}</td>
      <td><span class="badge ${a.risk_level === 'HIGH' || a.risk_level === 'CRITICAL' ? 'badge-high' : 'badge-low'}">${escapeHtml(a.risk_level)} • ${a.risk_pct}%</span></td>
      <td class="mono">${escapeHtml(a.account_key)}</td>
      <td><b>${escapeHtml(a.customer_name)}</b><div style="font-size:11px; color:var(--text-muted);">${escapeHtml(a.bank_name)}</div></td>
      <td style="font-size:11.5px;">${escapeHtml(a.typology)}</td>
      <td style="font-weight:600;">${escapeHtml(a.amount)}</td>
      <td style="font-size:12px;">${a.transactions_count} flows</td>
      <td style="font-size:11.5px; color:var(--text-muted);">${escapeHtml(a.created_date.split(' ')[0])}</td>
      <td><span class="badge-status badge-${a.status.toLowerCase().replace(/\s+/g, '')}">${escapeHtml(a.status)}</span></td>
      <td><button class="btn btn-secondary btn-sm" onclick="event.stopPropagation(); inspectAlert('${a.account_key}')">Investigate</button></td>
    `;
    tr.addEventListener('click', () => inspectAlert(a.account_key));
    tbody.appendChild(tr);
  });
}

function filterOverviewAlerts() {
  const rFilter = document.getElementById('filterAlertRisk').value;
  const sFilter = document.getElementById('filterAlertStatus').value;

  let alerts = (appState.overviewData && appState.overviewData.priority_alerts) || [];
  if (rFilter !== 'ALL') {
    alerts = alerts.filter(a => a.risk_level === rFilter);
  }
  if (sFilter !== 'ALL') {
    alerts = alerts.filter(a => a.status === sFilter);
  }
  renderOverviewAlertsTable(alerts);
}

async function inspectAlert(accountKey) {
  await loadCase(accountKey);
  switchView('case');
}

// -----------------------------------------------------------------------------
// Full Alerts Queue Tab
// -----------------------------------------------------------------------------

async function refreshAlerts() {
  const resp = await fetch('/api/alerts?limit=50');
  if (!resp.ok) return;
  const data = await resp.json();
  appState.alertsData = data.alerts || [];

  const tbody = document.getElementById('tbodyFullAlerts');
  if (!tbody) return;
  tbody.innerHTML = '';

  appState.alertsData.forEach(a => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="mono" style="font-weight:600; color:var(--primary-blue); cursor:pointer;" onclick="inspectAlert('${a.account_key}')">${escapeHtml(a.alert_id)}</td>
      <td><span class="badge ${a.risk_level === 'HIGH' || a.risk_level === 'CRITICAL' ? 'badge-high' : 'badge-low'}">${escapeHtml(a.risk_level)}</span></td>
      <td class="mono">${escapeHtml(a.account_key)}</td>
      <td><b>${escapeHtml(a.customer_name)}</b></td>
      <td style="font-size:11.5px;">${escapeHtml(a.typology)}</td>
      <td><b>${escapeHtml(a.amount)}</b></td>
      <td style="font-size:12px;">${a.transactions_count}</td>
      <td><span class="badge-status badge-${a.status.toLowerCase().replace(/\s+/g, '')}">${escapeHtml(a.status)}</span></td>
      <td><button class="btn btn-secondary btn-sm" onclick="inspectAlert('${a.account_key}')">Investigate</button></td>
    `;
    tbody.appendChild(tr);
  });
}

// -----------------------------------------------------------------------------
// Audit Log Tab
// -----------------------------------------------------------------------------

async function refreshAuditLog() {
  const resp = await fetch('/api/audit_log?limit=50');
  if (!resp.ok) return;
  const logs = await resp.json();
  appState.auditLogData = logs;

  const tbody = document.getElementById('tbodyAuditLog');
  if (!tbody) return;
  tbody.innerHTML = '';

  logs.forEach(l => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="mono" style="font-size:11.5px; color:var(--text-muted);">${escapeHtml(l.timestamp)}</td>
      <td><b>${escapeHtml(l.investigator)}</b></td>
      <td><span class="badge badge-low" style="font-weight:600;">${escapeHtml(l.action)}</span></td>
      <td class="mono" style="color:var(--primary-blue);">${escapeHtml(l.case_id)}</td>
      <td style="font-size:12.5px;">${escapeHtml(l.details)}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function logAuditAction(action, caseId, details) {
  try {
    await fetch('/api/audit_log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        action,
        case_id: caseId || 'AML-GENERAL',
        details,
        investigator: 'H. Vance (Senior AML Analyst)'
      })
    });
    await refreshAuditLog();
  } catch (err) {
    console.error('[TRACE] Failed to log audit event:', err);
  }
}

// -----------------------------------------------------------------------------
// Model & Benchmarks Tab
// -----------------------------------------------------------------------------

async function refreshBenchmarks() {
  const resp = await fetch('/api/benchmarks');
  if (!resp.ok) return;
  appState.benchmarksData = await resp.json();
}

// -----------------------------------------------------------------------------
// Side Detail Panel
// -----------------------------------------------------------------------------

function openSidePanel(txId) {
  const tx = appState.activeTransactions.find(t => t.id === txId);
  const panel = document.getElementById('sideDetailPanel');
  const titleEl = document.getElementById('sideDetailTitle');
  const bodyEl = document.getElementById('sideDetailBody');

  if (!panel || !titleEl || !bodyEl) return;

  if (tx) {
    titleEl.innerText = `Transaction ${tx.id}`;
    bodyEl.innerHTML = `
      <div style="display:flex; flex-direction:column; gap:12px;">
        <div style="background:var(--surface-secondary); padding:12px; border-radius:var(--radius-sm); border:1px solid var(--border-subtle);">
          <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase;">Observed Transfer Amount</div>
          <div style="font-size:20px; font-weight:700; color:var(--text-primary); margin-top:2px;">${escapeHtml(tx.amount)}</div>
          <div style="margin-top:6px;"><span class="badge ${tx.risk === 'HIGH' ? 'badge-high' : 'badge-low'}">${escapeHtml(tx.risk)} RISK CONVICTION</span></div>
        </div>

        <div style="font-size:12.5px; display:flex; flex-direction:column; gap:6px;">
          <div>Timestamp: <b>${escapeHtml(tx.date)}</b></div>
          <div>Format / Type: <b>${escapeHtml(tx.type)}</b></div>
          <div>Direction: <b>${escapeHtml(tx.direction)}</b></div>
          <div>Source Entity: <b class="mono">${escapeHtml(tx.from)}</b> (${escapeHtml(tx.from_customer || 'Bank Client')})</div>
          <div>Destination Entity: <b class="mono">${escapeHtml(tx.to)}</b> (${escapeHtml(tx.to_customer || 'Bank Client')})</div>
        </div>

        <div style="border-top:1px solid var(--border-subtle); padding-top:10px; margin-top:4px;">
          <button class="btn btn-primary btn-sm" style="width:100%;" onclick="toggleTxnSelection('${tx.id}'); closeSidePanel();">
            ${appState.selectedTxns.has(tx.id) ? 'Deselect Transaction' : 'Select for Evidentiary Docket'}
          </button>
        </div>
      </div>
    `;
  } else {
    titleEl.innerText = `Entity Context`;
    bodyEl.innerHTML = `<div style="font-size:12.5px; color:var(--text-secondary);">Inspecting related network entity.</div>`;
  }

  panel.classList.add('open');
}

function closeSidePanel() {
  const panel = document.getElementById('sideDetailPanel');
  if (panel) panel.classList.remove('open');
}

// -----------------------------------------------------------------------------
// Escalation & Status Modals
// -----------------------------------------------------------------------------

function initModals() {
  const btnOpenEscalate = document.getElementById('btnOpenEscalateModal');
  const modalEscalate = document.getElementById('modalEscalate');
  const btnConfirmEscalate = document.getElementById('btnConfirmEscalate');

  if (btnOpenEscalate && modalEscalate) {
    btnOpenEscalate.addEventListener('click', () => {
      modalEscalate.classList.add('open');
    });
  }

  if (btnConfirmEscalate && modalEscalate) {
    btnConfirmEscalate.addEventListener('click', async () => {
      const reason = document.getElementById('modalEscalateReason').value;
      const notes = document.getElementById('modalEscalateNotes').value;
      
      try {
        const resp = await fetch(`/api/cases/${encodeURIComponent(appState.activeCaseId)}/status`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            status: 'ESCALATED',
            reason: `${reason}${notes ? ' - ' + notes : ''}`,
            investigator: 'H. Vance (Senior AML Analyst)'
          })
        });

        if (resp.ok) {
          modalEscalate.classList.remove('open');
          await loadCase(appState.activeCaseId);
          alert(`Case ${appState.activeCaseId} escalated to Compliance Review Committee.`);
        }
      } catch (err) {
        console.error('[TRACE] Failed to escalate case:', err);
      }
    });
  }
}

function closeEscalateModal() {
  const modal = document.getElementById('modalEscalate');
  if (modal) modal.classList.remove('open');
}

async function closeCase() {
  if (!confirm(`Confirm closure of Case ${appState.activeCaseId}?`)) return;

  try {
    const resp = await fetch(`/api/cases/${encodeURIComponent(appState.activeCaseId)}/status`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        status: 'CLOSED — NO FURTHER ACTION',
        reason: 'Investigation complete. Low residual risk confirmed.',
        investigator: 'H. Vance (Senior AML Analyst)'
      })
    });

    if (resp.ok) {
      await loadCase(appState.activeCaseId);
      alert(`Case ${appState.activeCaseId} closed with disposition: NO FURTHER ACTION.`);
    }
  } catch (err) {
    console.error('[TRACE] Failed to close case:', err);
  }
}

// -----------------------------------------------------------------------------
// Search & Keyboard Shortcuts
// -----------------------------------------------------------------------------

async function handleGlobalSearch(query) {
  try {
    // Check if matching case 1, 2, or 3
    if (query.includes('496618') || query.includes('80C1C8AC0') || query.includes('421')) {
      await loadCase('case1');
      switchView('case');
      return;
    }
    if (query.includes('6') || query.includes('8000EBAC0') || query.includes('389')) {
      await loadCase('case2');
      switchView('case');
      return;
    }
    if (query.includes('91') || query.includes('800056370') || query.includes('54')) {
      await loadCase('case3');
      switchView('case');
      return;
    }

    // Try direct account query
    const resp = await fetch(`/api/accounts/${encodeURIComponent(query)}`);
    if (resp.ok) {
      await loadCase(query);
      switchView('case');
    } else {
      alert(`Account key or identifier "${query}" not found in graph index.`);
    }
  } catch (err) {
    console.error('[TRACE] Search error:', err);
  }
}

function initKeyboardShortcuts() {
  window.addEventListener('keydown', (e) => {
    // If typing in input or textarea, don't trigger shortcuts
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    if (e.key === 'p' || e.key === 'P') {
      togglePresentationMode();
    } else if (e.key === '1') {
      loadCase('case1');
      switchView('case');
    } else if (e.key === '2') {
      loadCase('case2');
      switchView('case');
    } else if (e.key === '3') {
      loadCase('case3');
      switchView('case');
    } else if (e.key === 'r' || e.key === 'R') {
      fitGraphToViewport();
    } else if (e.key === 'e' || e.key === 'E') {
      switchView('case');
      switchCaseTab('network');
      toggleTechAccordion();
    }
  });
}

function togglePresentationMode() {
  appState.presentationMode = !appState.presentationMode;
  document.body.classList.toggle('presentation-mode', appState.presentationMode);
  setTimeout(fitGraphToViewport, 100);
}

function toggleTechAccordion() {
  const body = document.getElementById('techAccordionBody');
  const arrow = document.getElementById('techAccordionArrow');
  if (!body) return;

  const isHidden = body.style.display === 'none' || !body.style.display;
  body.style.display = isHidden ? 'block' : 'none';
  if (arrow) arrow.innerHTML = isHidden ? '&#9662;' : '&#9656;';
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

function setText(elementId, text) {
  const el = document.getElementById(elementId);
  if (el) el.innerText = text;
}

function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
