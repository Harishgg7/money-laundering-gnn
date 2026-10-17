/**
 * Explainable AML GNN — Client-side application
 * Minimal, clean graph visualizer, benchmarks sandbox, and audit memo.
 */

const state = {
  activeTab: 'explorer',
  activeNodeId: 496618,
  network: {
    nodes: [],
    edges: [],
    camera: { x: 0, y: 0, zoom: 1 },
    draggedNode: null,
    isDragging: false,
    dragStart: { x: 0, y: 0 },
    hoveredNode: null,
  },
  flaggedAccounts: [],
  benchmarkData: null,
  currentExplanation: null,
};

const dom = {
  canvas: document.getElementById('cleanCanvas'),
  ctx: null,
  tabs: document.querySelectorAll('.tab-btn'),
  panes: document.querySelectorAll('.tab-pane'),
  caseButtons: document.querySelectorAll('.case-btn'),
  flaggedList: document.getElementById('flaggedTableBody'),
  customInput: document.getElementById('customNodeInput'),
  btnSearch: document.getElementById('btnSearchNode'),
  btnReset: document.getElementById('btnResetView'),
  btnZoomIn: document.getElementById('btnZoomIn'),
  btnZoomOut: document.getElementById('btnZoomOut'),
  // Dossier elements
  dossierKey: document.getElementById('dossierTargetKey'),
  dossierRisk: document.getElementById('dossierTargetRisk'),
  dossierRole: document.getElementById('dossierTargetRole'),
  dossierInDeg: document.getElementById('dossierInDeg'),
  dossierOutDeg: document.getElementById('dossierOutDeg'),
  dossierSummary: document.getElementById('dossierSummary'),
  dossierEdgeList: document.getElementById('dossierEdgeList'),
  // Sandbox elements
  thSlider: document.getElementById('thSlider'),
  thDisplay: document.getElementById('thDisplay'),
  sbPrecision: document.getElementById('sbPrecision'),
  sbRecall: document.getElementById('sbRecall'),
  sbF1: document.getElementById('sbF1'),
  sbAlerts: document.getElementById('sbAlerts'),
  sbFalseAlarms: document.getElementById('sbFalseAlarms'),
  sbTruePositives: document.getElementById('sbTruePositives'),
  // SAR elements
  sarTracking: document.getElementById('sarTrackingId'),
  sarSubjectKey: document.getElementById('sarSubjectKey'),
  sarSubjectBank: document.getElementById('sarSubjectBank'),
  sarSubjectRisk: document.getElementById('sarSubjectRisk'),
  sarTimestamp: document.getElementById('sarTimestamp'),
  sarNarrative: document.getElementById('sarNarrativeText'),
  sarEvidenceTable: document.getElementById('sarEvidenceTableBody'),
  sarRecommendations: document.getElementById('sarRecommendationsList'),
  btnCopySar: document.getElementById('btnCopySar'),
  btnDownloadSar: document.getElementById('btnDownloadSar'),
  btnPrintSar: document.getElementById('btnPrintSar'),
};

// -----------------------------------------------------------------------------
// Initialization
// -----------------------------------------------------------------------------
window.addEventListener('DOMContentLoaded', async () => {
  // Ensure white theme alone and clear any legacy saved theme preference
  localStorage.removeItem('aml_theme');
  document.documentElement.removeAttribute('data-theme');

  if (dom.canvas) {
    dom.ctx = dom.canvas.getContext('2d');
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    setupCanvasInteractions();
  }

  setupNavigation();
  setupEventListeners();

  await loadSystemStats();
  await loadFlaggedAccounts();
  await loadBenchmarks();
  await loadAccountExplanation(state.activeNodeId);

  // Render loop
  requestAnimationFrame(drawGraph);
});

// -----------------------------------------------------------------------------
// Canvas Resizing & Camera
// -----------------------------------------------------------------------------
function resizeCanvas() {
  if (!dom.canvas) return;
  const parent = dom.canvas.parentElement;
  dom.canvas.width = parent.clientWidth * window.devicePixelRatio;
  dom.canvas.height = parent.clientHeight * window.devicePixelRatio;
  dom.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
  if (state.network.camera.x === 0 && state.network.camera.y === 0) {
    state.network.camera.x = parent.clientWidth / 2;
    state.network.camera.y = parent.clientHeight / 2;
  }
}

// -----------------------------------------------------------------------------
// Navigation Tabs
// -----------------------------------------------------------------------------
function setupNavigation() {
  dom.tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      dom.tabs.forEach(t => t.classList.remove('active'));
      dom.panes.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      document.getElementById(`pane-${target}`).classList.add('active');
      state.activeTab = target;

      if (target === 'explorer') {
        setTimeout(resizeCanvas, 40);
      } else if (target === 'sar') {
        updateSarMemo();
      }
    });
  });
}

// -----------------------------------------------------------------------------
// API Calls
// -----------------------------------------------------------------------------
async function loadSystemStats() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    document.getElementById('statTotalAccounts').textContent = data.total_accounts.toLocaleString();
    document.getElementById('statTotalEdges').textContent = data.total_edges.toLocaleString();
    document.getElementById('statRawTxns').textContent = (data.raw_transactions_ingested / 1e6).toFixed(2) + 'M';
  } catch (err) {
    console.warn('System status not available:', err);
  }
}

async function loadFlaggedAccounts() {
  try {
    const res = await fetch('/api/accounts/flagged?limit=25');
    const accounts = await res.json();
    state.flaggedAccounts = accounts;
    renderFlaggedList(accounts);
  } catch (err) {
    console.warn('Flagged accounts fetch failed:', err);
  }
}

function renderFlaggedList(accounts) {
  if (!dom.flaggedList) return;
  dom.flaggedList.innerHTML = '';
  accounts.forEach(acc => {
    const item = document.createElement('div');
    item.className = `flagged-item ${acc.node_id === state.activeNodeId ? 'active' : ''}`;
    item.innerHTML = `
      <div>
        <div class="flagged-key">${acc.account_key}</div>
        <div style="font-size:0.7rem; color:var(--ink-muted)">Node #${acc.node_id} · ${acc.typology_tag}</div>
      </div>
      <div class="flagged-prob">${(acc.fraud_probability * 100).toFixed(1)}%</div>
    `;
    item.addEventListener('click', () => {
      document.querySelectorAll('.flagged-item').forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      loadAccountExplanation(acc.node_id);
    });
    dom.flaggedList.appendChild(item);
  });
}

async function loadAccountExplanation(nodeId) {
  try {
    const res = await fetch(`/api/explain/${nodeId}`);
    if (!res.ok) throw new Error(`Node ${nodeId} error`);
    const data = await res.json();
    state.activeNodeId = nodeId;
    state.currentExplanation = data;

    updateDossier(data);
    buildCleanGraph(data);
    updateSarMemo();
  } catch (err) {
    console.error('Explanation error:', err);
  }
}

function updateDossier(data) {
  if (dom.dossierKey) dom.dossierKey.textContent = data.account_key;
  if (dom.dossierRisk) dom.dossierRisk.textContent = `${(data.risk_probability * 100).toFixed(1)}% Risk`;
  if (dom.dossierSummary) dom.dossierSummary.textContent = data.summary;

  const targetNode = data.nodes.find(n => n.id === data.node_id) || data.nodes[0];
  if (targetNode) {
    if (dom.dossierRole) dom.dossierRole.textContent = targetNode.role;
    if (dom.dossierInDeg) dom.dossierInDeg.textContent = targetNode.in_degree;
    if (dom.dossierOutDeg) dom.dossierOutDeg.textContent = targetNode.out_degree;
  }

  // Decisive flows table
  if (dom.dossierEdgeList) {
    dom.dossierEdgeList.innerHTML = '';
    data.edges.slice(0, 7).forEach(e => {
      const tr = document.createElement('tr');
      const src = e.source_key.split('-')[1] || e.source_key;
      const dst = e.target_key.split('-')[1] || e.target_key;
      tr.innerHTML = `
        <td>
          <span style="color:${e.is_decisive ? 'var(--danger)' : 'var(--ink)'}; font-weight:${e.is_decisive ? '600' : '400'}">${src}</span>
          <span style="color:var(--ink-muted)">➔</span>
          <span>${dst}</span>
        </td>
        <td style="text-align:right; font-weight:600; color:${e.is_decisive ? 'var(--danger)' : 'var(--ink)'}">${e.importance.toFixed(4)}</td>
      `;
      dom.dossierEdgeList.appendChild(tr);
    });
  }
}

// -----------------------------------------------------------------------------
// Clean Network Graph Construction & Drawing
// -----------------------------------------------------------------------------
function buildCleanGraph(data) {
  const width = dom.canvas.parentElement.clientWidth;
  const height = dom.canvas.parentElement.clientHeight;
  const cx = width / 2;
  const cy = height / 2;

  const nodes = [];
  const edges = [];
  const nodeMap = new Map();

  const total = data.nodes.length;
  data.nodes.forEach((n, idx) => {
    let x, y;
    if (n.id === data.node_id) {
      // Flagged target placed cleanly on right-center
      x = cx + 130;
      y = cy;
    } else if (n.role.includes('Hub') || n.role.includes('Intermediary')) {
      x = cx - 30;
      y = cy + (idx % 2 === 0 ? 40 : -40);
    } else {
      const angle = (idx / Math.max(1, total - 1)) * Math.PI * 1.3 - Math.PI * 0.65;
      const r = 210 + (idx % 3) * 25;
      x = cx - r * Math.cos(angle);
      y = cy + r * Math.sin(angle);
    }

    const nodeObj = {
      id: n.id,
      key: n.key,
      role: n.role,
      risk_score: n.risk_score,
      is_target: n.is_target,
      in_degree: n.in_degree,
      out_degree: n.out_degree,
      x: x,
      y: y,
      radius: n.is_target ? 16 : (n.role.includes('Hub') ? 13 : 9),
      color: getNodeColor(n),
    };
    nodes.push(nodeObj);
    nodeMap.set(n.id, nodeObj);
  });

  data.edges.forEach(e => {
    const src = nodeMap.get(e.source);
    const dst = nodeMap.get(e.target);
    if (src && dst) {
      edges.push({
        source: src,
        target: dst,
        source_key: e.source_key,
        target_key: e.target_key,
        importance: e.importance,
        is_decisive: e.is_decisive,
        is_self_loop: e.is_self_loop,
      });
    }
  });

  state.network.nodes = nodes;
  state.network.edges = edges;
  state.network.camera.zoom = 1;
  state.network.camera.x = 0;
  state.network.camera.y = 0;
}

function getNodeColor(node) {
  if (node.is_target) return '#dc2626'; // Deep Red
  if (node.role.includes('Hub')) return '#d97706'; // Amber
  if (node.role.includes('Mule') || node.role.includes('Source')) return '#0c8592'; // Deep Teal
  return '#64748b'; // Slate
}

function drawGraph() {
  const ctx = dom.ctx;
  if (!ctx || state.activeTab !== 'explorer') {
    requestAnimationFrame(drawGraph);
    return;
  }

  const width = dom.canvas.parentElement.clientWidth;
  const height = dom.canvas.parentElement.clientHeight;

  ctx.clearRect(0, 0, width, height);
  ctx.save();

  // Subtle background grid
  drawSubtleGrid(ctx, width, height);

  // Apply Camera transform
  ctx.translate(width / 2 + state.network.camera.x, height / 2 + state.network.camera.y);
  ctx.scale(state.network.camera.zoom, state.network.camera.zoom);
  ctx.translate(-width / 2, -height / 2);

  // Draw Edges
  state.network.edges.forEach(e => {
    if (e.is_self_loop) return;

    const s = e.source;
    const d = e.target;
    const dx = d.x - s.x;
    const dy = d.y - s.y;
    const midX = (s.x + d.x) / 2 - dy * 0.08;
    const midY = (s.y + d.y) / 2 + dx * 0.08;

    ctx.beginPath();
    ctx.moveTo(s.x, s.y);
    ctx.quadraticCurveTo(midX, midY, d.x, d.y);

    if (e.is_decisive) {
      ctx.strokeStyle = '#dc2626';
      ctx.lineWidth = 3.5;
    } else {
      ctx.strokeStyle = 'rgba(100, 116, 139, 0.45)';
      ctx.lineWidth = Math.max(1, e.importance * 3);
    }
    ctx.stroke();

    // Clean Arrowhead
    drawCleanArrow(ctx, midX, midY, d.x, d.y, d.radius, e.is_decisive ? '#dc2626' : '#64748b');

    // Draw numerical weight on decisive edge
    if (e.is_decisive) {
      ctx.font = '600 10px "IBM Plex Mono", monospace';
      ctx.fillStyle = '#dc2626';
      ctx.textAlign = 'center';
      ctx.fillText(`w=${e.importance.toFixed(3)}`, midX, midY - 6);
    }
  });

  // Draw Nodes
  state.network.nodes.forEach(n => {
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
    ctx.fillStyle = n.color;
    ctx.fill();

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Node label
    ctx.font = `${n.is_target ? '600 11px' : '400 9px'} "IBM Plex Mono", monospace`;
    ctx.fillStyle = '#0f172a';
    ctx.textAlign = 'center';
    const label = n.key.split('-')[1] || n.key;
    ctx.fillText(label, n.x, n.y + n.radius + 13);

    if (n.is_target) {
      ctx.font = '600 8.5px "IBM Plex Sans", sans-serif';
      ctx.fillStyle = '#dc2626';
      ctx.fillText('TARGET', n.x, n.y - n.radius - 5);
    } else if (n.role.includes('Hub')) {
      ctx.font = '600 8px "IBM Plex Sans", sans-serif';
      ctx.fillStyle = '#d97706';
      ctx.fillText('HUB', n.x, n.y - n.radius - 4);
    }
  });

  ctx.restore();
  requestAnimationFrame(drawGraph);
}

function drawSubtleGrid(ctx, width, height) {
  ctx.save();
  ctx.fillStyle = '#e2e8f0';
  const step = 32;
  for (let x = 16; x < width; x += step) {
    for (let y = 16; y < height; y += step) {
      ctx.fillRect(x, y, 1, 1);
    }
  }
  ctx.restore();
}

function drawCleanArrow(ctx, fromX, fromY, toX, toY, radius, color) {
  const dx = toX - fromX;
  const dy = toY - fromY;
  const angle = Math.atan2(dy, dx);
  const headLen = 7;

  const tipX = toX - Math.cos(angle) * (radius + 2);
  const tipY = toY - Math.sin(angle) * (radius + 2);

  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(tipX - headLen * Math.cos(angle - Math.PI / 6), tipY - headLen * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(tipX - headLen * Math.cos(angle + Math.PI / 6), tipY - headLen * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fillStyle = color;
  ctx.fill();
}

// -----------------------------------------------------------------------------
// Mouse & Canvas Drag Interactions
// -----------------------------------------------------------------------------
function setupCanvasInteractions() {
  const canvas = dom.canvas;

  canvas.addEventListener('mousedown', (e) => {
    const coords = getCanvasCoords(e);
    const clicked = state.network.nodes.find(n => Math.hypot(n.x - coords.x, n.y - coords.y) <= n.radius + 6);
    if (clicked) {
      state.network.draggedNode = clicked;
      updateDossierFromNode(clicked);
    } else {
      state.network.isDragging = true;
      state.network.dragStart = { x: e.clientX, y: e.clientY };
    }
  });

  window.addEventListener('mousemove', (e) => {
    if (state.network.draggedNode) {
      const coords = getCanvasCoords(e);
      state.network.draggedNode.x = coords.x;
      state.network.draggedNode.y = coords.y;
    } else if (state.network.isDragging) {
      state.network.camera.x += e.clientX - state.network.dragStart.x;
      state.network.camera.y += e.clientY - state.network.dragStart.y;
      state.network.dragStart = { x: e.clientX, y: e.clientY };
    }
  });

  window.addEventListener('mouseup', () => {
    state.network.draggedNode = null;
    state.network.isDragging = false;
  });

  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.08 : 0.92;
    state.network.camera.zoom = Math.min(2.2, Math.max(0.5, state.network.camera.zoom * factor));
  });
}

function getCanvasCoords(e) {
  const rect = dom.canvas.getBoundingClientRect();
  const rawX = e.clientX - rect.left;
  const rawY = e.clientY - rect.top;
  const width = rect.width;
  const height = rect.height;

  const centeredX = rawX - (width / 2 + state.network.camera.x);
  const centeredY = rawY - (height / 2 + state.network.camera.y);
  return {
    x: centeredX / state.network.camera.zoom + width / 2,
    y: centeredY / state.network.camera.zoom + height / 2,
  };
}

function updateDossierFromNode(node) {
  if (dom.dossierKey) dom.dossierKey.textContent = node.key;
  if (dom.dossierRisk) dom.dossierRisk.textContent = `${(node.risk_score * 100).toFixed(1)}% Risk`;
  if (dom.dossierRole) dom.dossierRole.textContent = node.role;
  if (dom.dossierInDeg) dom.dossierInDeg.textContent = node.in_degree;
  if (dom.dossierOutDeg) dom.dossierOutDeg.textContent = node.out_degree;
}

// -----------------------------------------------------------------------------
// UI Event Handlers
// -----------------------------------------------------------------------------
function setupEventListeners() {
  // Camera Buttons
  if (dom.btnReset) {
    dom.btnReset.addEventListener('click', () => {
      state.network.camera.x = 0;
      state.network.camera.y = 0;
      state.network.camera.zoom = 1;
    });
  }
  if (dom.btnZoomIn) {
    dom.btnZoomIn.addEventListener('click', () => {
      state.network.camera.zoom = Math.min(2.2, state.network.camera.zoom * 1.2);
    });
  }
  if (dom.btnZoomOut) {
    dom.btnZoomOut.addEventListener('click', () => {
      state.network.camera.zoom = Math.max(0.5, state.network.camera.zoom * 0.8);
    });
  }

  // Case Preset Buttons
  dom.caseButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      dom.caseButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const nodeId = parseInt(btn.dataset.nodeId);
      loadAccountExplanation(nodeId);
    });
  });

  // Custom Search
  if (dom.btnSearch) {
    dom.btnSearch.addEventListener('click', () => {
      const val = parseInt(dom.customInput.value.trim());
      if (!isNaN(val)) {
        loadAccountExplanation(val);
      }
    });
  }

  // Threshold Slider
  if (dom.thSlider) {
    dom.thSlider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      dom.thDisplay.textContent = val.toFixed(2);
      updateThresholdMetrics(val);
    });
  }

  // SAR Actions
  if (dom.btnPrintSar) dom.btnPrintSar.addEventListener('click', () => window.print());
  if (dom.btnCopySar) dom.btnCopySar.addEventListener('click', copySarText);
  if (dom.btnDownloadSar) dom.btnDownloadSar.addEventListener('click', downloadSarJson);
}

// -----------------------------------------------------------------------------
// Benchmark Sandbox
// -----------------------------------------------------------------------------
async function loadBenchmarks() {
  try {
    const res = await fetch('/api/benchmarks');
    const data = await res.json();
    state.benchmarkData = data;
    updateThresholdMetrics(0.68);
  } catch (err) {
    console.warn('Failed to load benchmarks:', err);
  }
}

function updateThresholdMetrics(val) {
  if (!state.benchmarkData || !state.benchmarkData.threshold_curve) return;
  const curve = state.benchmarkData.threshold_curve;

  let closest = curve[0];
  let minDiff = 999;
  curve.forEach(pt => {
    const diff = Math.abs(pt.threshold - val);
    if (diff < minDiff) {
      minDiff = diff;
      closest = pt;
    }
  });

  dom.sbPrecision.textContent = (closest.precision * 100).toFixed(1) + '%';
  dom.sbRecall.textContent = (closest.recall * 100).toFixed(1) + '%';
  dom.sbF1.textContent = closest.f1.toFixed(3);
  dom.sbAlerts.textContent = closest.estimated_alerts.toLocaleString();
  dom.sbFalseAlarms.textContent = closest.false_alarms.toLocaleString();
  dom.sbTruePositives.textContent = closest.true_positives.toLocaleString();
}

// -----------------------------------------------------------------------------
// SAR Audit Memo
// -----------------------------------------------------------------------------
async function updateSarMemo() {
  try {
    const res = await fetch(`/api/sar/${state.activeNodeId}`);
    if (!res.ok) return;
    const sar = await res.json();

    if (dom.sarTracking) dom.sarTracking.textContent = sar.sar_tracking_number;
    if (dom.sarSubjectKey) dom.sarSubjectKey.textContent = sar.subject_entity.composite_key;
    if (dom.sarSubjectBank) dom.sarSubjectBank.textContent = sar.subject_entity.financial_institution_code;
    if (dom.sarSubjectRisk) dom.sarSubjectRisk.textContent = `${sar.subject_entity.assessed_fraud_probability} (${sar.subject_entity.risk_classification})`;
    if (dom.sarTimestamp) dom.sarTimestamp.textContent = sar.timestamp_generated;
    if (dom.sarNarrative) dom.sarNarrative.textContent = sar.executive_narrative;

    if (dom.sarEvidenceTable) {
      dom.sarEvidenceTable.innerHTML = '';
      sar.evidentiary_transactions.forEach(ev => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${ev.flow}</td>
          <td style="color:${ev.confidence_tier === 'CRITICAL' ? 'var(--danger)' : 'var(--ink)'}; font-weight:600;">${ev.attribution_score}</td>
          <td>${ev.relationship}</td>
        `;
        dom.sarEvidenceTable.appendChild(tr);
      });
    }

    if (dom.sarRecommendations) {
      dom.sarRecommendations.innerHTML = '';
      sar.statutory_recommendations.forEach(rec => {
        const li = document.createElement('li');
        li.textContent = rec;
        dom.sarRecommendations.appendChild(li);
      });
    }
  } catch (err) {
    console.warn('SAR fetch error:', err);
  }
}

async function copySarText() {
  try {
    const res = await fetch(`/api/sar/${state.activeNodeId}`);
    const data = await res.json();
    await navigator.clipboard.writeText(data.executive_narrative);
    alert('Audit narrative copied to clipboard.');
  } catch (err) {
    alert('Failed to copy.');
  }
}

async function downloadSarJson() {
  try {
    const res = await fetch(`/api/sar/${state.activeNodeId}`);
    const data = await res.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `SAR_Node_${state.activeNodeId}.json`;
    a.click();
  } catch (err) {
    alert('Failed to download JSON.');
  }
}
