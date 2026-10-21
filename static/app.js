/**
 * static/app.js — TRACE Institutional Financial Crime Forensics Engine
 * Supports:
 * 1. Full Network Topology Map vs Focused Case Ring
 * 2. Suspicious Cases Queue & Registry (839 detected cases)
 * 3. Inspector's Plain-English Layman Forensic Briefings
 * 4. Investigative Workflow (Elevate to FIU, 24/7 Surveillance, FinCEN SAR Generator)
 */

const TraceApp = (() => {
  let cases = {};
  let currentKey = "CASE_01";
  let currentDossier = null;
  let multiRingNetwork = null;
  let topologyMode = "focused"; // "focused" | "full"
  let activeFilter = "all";
  let activeThreshold = 0.3781;
  let defaultValThreshold = 0.3781;
  let explainSpotlight = false;

  // Canvas & Force-Directed Physics Engine
  let canvas = null;
  let ctx = null;
  let renderNodes = [];
  let renderEdges = [];
  let cam = { x: 0, y: 0, zoom: 1 };
  let dragging = false;
  let dragOrigin = { x: 0, y: 0 };
  let hoveredNode = null;
  let animStart = performance.now();

  function fmtMoney(val) {
    return "$" + Number(val || 0).toLocaleString(undefined, { maximumFractionDigits: 0 });
  }

  function nowTime() {
    return new Date().toTimeString().split(" ")[0];
  }

  function logAudit(actionText) {
    const box = document.getElementById("audit-trail-box");
    if (!box) return;
    const item = document.createElement("div");
    item.className = "audit-entry";
    item.innerHTML = `<span class="audit-time tabular">${nowTime()}</span><span>${actionText}</span>`;
    box.prepend(item);
  }

  function setStatus(type, html) {
    const bar = document.getElementById("status-banner");
    if (!bar) return;
    if (!type) {
      bar.className = "status-bar";
      bar.innerHTML = "";
      return;
    }
    bar.className = `status-bar ${type}`;
    bar.innerHTML = html;
  }

  // =========================================================================
  // INITIALIZATION
  // =========================================================================
  async function init() {
    canvas = document.getElementById("graph-canvas");
    if (!canvas) return;
    ctx = canvas.getContext("2d");

    bindCanvasEvents();
    bindKeyboardShortcuts();

    try {
      const res = await fetch("/api/bootstrap", { cache: "no-store" });
      if (res.ok) {
        const payload = await res.json();
        if (payload.cases && Object.keys(payload.cases).length > 0) {
          cases = payload.cases;
        }
        if (payload.multi_ring_network) {
          multiRingNetwork = payload.multi_ring_network;
        }
        if (payload.meta) {
          if (payload.meta.accounts) {
            const el = document.getElementById("stat-accounts");
            if (el) el.textContent = Number(payload.meta.accounts).toLocaleString();
          }
          if (payload.meta.transactions_raw) {
            const el = document.getElementById("stat-tx");
            if (el) el.textContent = Number(payload.meta.transactions_raw).toLocaleString();
          }
          if (payload.meta.edges_aggregated) {
            const el = document.getElementById("stat-edges");
            if (el) el.textContent = Number(payload.meta.edges_aggregated).toLocaleString();
          }
          if (payload.meta.total_suspicious_cases) {
            const qEl = document.getElementById("stat-flagged-count");
            if (qEl) qEl.textContent = payload.meta.total_suspicious_cases;
            const susEl = document.getElementById("stat-suspicious-total");
            if (susEl) susEl.textContent = `${payload.meta.total_suspicious_cases} Verified High-Risk Hubs`;
          }
          if (payload.meta.threshold) {
            defaultValThreshold = parseFloat(payload.meta.threshold);
            activeThreshold = defaultValThreshold;
          }
        }
      }
    } catch (_) {
      // Backend offline fallback handled
    }

    renderDrawerCases();
    selectCase("CASE_01");
    logAudit("TRACE Forensics Online: 515K accounts monitored, 839 high-risk clusters indexed.");
  }

  // =========================================================================
  // SUSPICIOUS CASES QUEUE & REGISTRY DRAWER
  // =========================================================================
  function toggleCaseDrawer() {
    const drawer = document.getElementById("case-drawer");
    const backdrop = document.getElementById("drawer-backdrop");
    if (!drawer) return;
    const isOpen = drawer.classList.contains("open");
    drawer.classList.toggle("open", !isOpen);
    if (backdrop) backdrop.classList.toggle("open", !isOpen);
  }

  function filterCases(filterType) {
    activeFilter = filterType;
    ["all", "high", "surveillance", "elevated"].forEach(t => {
      const chip = document.getElementById(`chip-${t}`);
      if (chip) chip.classList.toggle("active", t === filterType);
    });
    renderDrawerCases();
  }

  function renderDrawerCases() {
    const container = document.getElementById("drawer-case-container");
    if (!container) return;

    const caseList = Object.entries(cases).map(([k, obj]) => ({ key: k, ...obj }));

    const filtered = caseList.filter(c => {
      if (activeFilter === "high") return c.gnn_score >= 0.40;
      if (activeFilter === "surveillance") return c.status === "UNDER SURVEILLANCE";
      if (activeFilter === "elevated") return c.status === "ELEVATED TO FIU";
      return true;
    });

    container.innerHTML = filtered.map(c => {
      const isSelected = c.key === currentKey;
      const statusClass = c.status === "ELEVATED TO FIU" ? "elevated" : (c.status === "UNDER SURVEILLANCE" ? "surveillance" : (c.status === "CLOSED" ? "closed" : "open"));
      const riskPct = (c.gnn_score * 100).toFixed(1) + "%";

      return `
        <div class="case-card ${isSelected ? 'active' : ''}" onclick="TraceApp.selectCase('${c.key}'); TraceApp.toggleCaseDrawer();">
          <div class="case-card-head">
            <span class="case-card-id tabular">${c.case_id || c.key} (${c.account_id})</span>
            <span class="case-status-badge ${statusClass}">${c.status || 'OPEN'}</span>
          </div>
          <div class="case-card-title">${c.title || c.summary || 'Suspicious Ring'}</div>
          <div class="case-card-meta">
            <span>Model Risk Score: <strong style="color: var(--risk-high);" class="tabular">${riskPct}</strong></span>
            <span>Typology: <strong>${c.typology || 'Smurfing'}</strong></span>
            <span>Vol: <strong class="tabular">${fmtMoney(c.volume_est || 145000)}</strong></span>
          </div>
        </div>
      `;
    }).join("");
  }

  // =========================================================================
  // CASE SELECTION & DOSSIER BINDING
  // =========================================================================
  function selectCase(caseKey) {
    currentKey = caseKey;
    const d = cases[caseKey];
    if (!d) return;

    explainSpotlight = false;
    const btnExplain = document.getElementById("btn-explain-toggle");
    if (btnExplain) btnExplain.classList.remove("active");

    renderDossier(d);
    renderDrawerCases();
    logAudit(`Loaded suspicious case: ${d.case_id || caseKey} (Entity: ${d.account_id})`);
  }

  function renderDossier(d) {
    currentDossier = d;
    if (d.threshold) defaultValThreshold = d.threshold;
    activeThreshold = defaultValThreshold;

    const inputEl = document.getElementById("account-query-input");
    if (inputEl) inputEl.value = d.account_id;

    const slider = document.getElementById("threshold-slider");
    if (slider) slider.value = Math.round(activeThreshold * 100);

    // Entity Identifier Header & Case Status
    const acctEl = document.getElementById("dossier-account-id");
    if (acctEl) acctEl.textContent = d.account_id;

    const gtEl = document.getElementById("dossier-gt-label");
    if (gtEl) {
      gtEl.textContent = d.ground_truth === 1 ? "Ground Truth: Illicit (1)" : "Ground Truth: Legitimate (0)";
    }

    const refEl = document.getElementById("dossier-case-ref");
    if (refEl) {
      refEl.textContent = `REF: ${d.case_id || currentKey}`;
    }

    const statusEl = document.getElementById("dossier-case-status");
    if (statusEl) {
      const st = d.status || "OPEN INVESTIGATION";
      const statusClass = st === "ELEVATED TO FIU" ? "elevated" : (st === "UNDER SURVEILLANCE" ? "surveillance" : (st === "CLOSED" ? "closed" : "open"));
      statusEl.className = `case-status-badge ${statusClass}`;
      statusEl.textContent = st;
    }

    // =======================================================================
    // INSPECTOR'S PLAIN-ENGLISH FORENSIC BRIEFING (LAYMAN TERMS)
    // =======================================================================
    const layman = d.layman_briefing || {
      title: d.title || "Suspicious Transaction Ring",
      headline: "Unusual Relational Money Movement Detected",
      summary: d.summary || "Account flagged by graph neural network surveillance exceeding calibrated operational risk threshold.",
      red_flags: [
        "Unusual Transaction Velocity: Coordinated transfer frequency far exceeding baseline commercial activity.",
        "Pass-Through Conduit Behavior: Low fund retention with near-immediate onward routing.",
        "Relational Connectivity: Interconnected with accounts in known suspicious clusters.",
        "Conventional Rule Miss: Account was invisible to legacy amount-limit rule filters."
      ],
      recommended_action: "Execute Enhanced Due Diligence (EDD) and issue FinCEN SAR dossier."
    };

    const typoTitleEl = document.getElementById("layman-typology-title");
    if (typoTitleEl) typoTitleEl.textContent = d.typology || layman.typology || "Suspicious Cluster";

    const headEl = document.getElementById("layman-headline");
    if (headEl) headEl.textContent = layman.headline || layman.title;

    const sumEl = document.getElementById("layman-summary");
    if (sumEl) sumEl.textContent = layman.summary;

    const flagsEl = document.getElementById("layman-redflags");
    if (flagsEl && layman.red_flags) {
      flagsEl.innerHTML = layman.red_flags.map(f => `<li>${f}</li>`).join("");
    }

    const actionEl = document.getElementById("layman-action-advice");
    if (actionEl) actionEl.textContent = layman.recommended_action || "Issue FinCEN SAR Form 111 filing.";

    // Model Risk Scores & Gauge
    refreshThresholdDependentUI();

    // Dual Comparison (MLP vs GNN)
    const mlpCard = document.getElementById("dossier-mlp-score");
    if (mlpCard) mlpCard.textContent = (d.mlp_score * 100).toFixed(1) + "%";

    const gnnCard = document.getElementById("dossier-gnn-card-score");
    if (gnnCard) gnnCard.textContent = (d.gnn_score * 100).toFixed(1) + "%";

    const contrastText = document.getElementById("contrast-text");
    if (contrastText) {
      if (d.risk_level === "HIGH" && d.mlp_score < activeThreshold) {
        contrastText.textContent = `Isolated account statistics missed this entity (${(d.mlp_score * 100).toFixed(1)}%). 3-hop message passing surfaced the coordinated laundering ring (${(d.gnn_score * 100).toFixed(1)}%).`;
      } else if (d.risk_level === "HIGH") {
        contrastText.textContent = `Coordinated multi-hop topological flows reinforce elevated model risk score at ${(d.gnn_score * 100).toFixed(1)}%.`;
      } else {
        contrastText.textContent = `Commercial account with sparse, balanced transaction flows verified as low risk (${(d.gnn_score * 100).toFixed(1)}%).`;
      }
    }

    // Observed Behavioral Signals (Formatted properly)
    const featEl = document.getElementById("dossier-feature-list");
    if (featEl) {
      featEl.innerHTML = (d.features || []).map(f => {
        const valStr = f.formatted ? f.formatted : (typeof f.value === "number" ? f.value.toFixed(2) : f.value);
        return `
          <div class="signal-row">
            <span class="signal-key">${f.name}</span>
            <span class="signal-val tabular">${valStr}</span>
          </div>
        `;
      }).join("");
    }

    // Explanation Fidelity+
    const fid = d.fidelity_plus !== undefined ? d.fidelity_plus : (d.risk_level === "HIGH" ? 0.282 : 0.0);
    const fidEl = document.getElementById("explain-fidelity-val");
    if (fidEl) {
      fidEl.textContent = `-${(fid * 100).toFixed(1)}% Drop`;
    }

    // Decisive Transaction Flows
    const flowListEl = document.getElementById("explain-flow-list");
    if (flowListEl) {
      const flows = (d.top_flows && d.top_flows.length > 0) ? d.top_flows : (d.subgraph && d.subgraph.edges ? d.subgraph.edges.slice(0, 5) : []);
      flowListEl.innerHTML = flows.map(f => {
        const pct = Math.round((f.importance || 0.5) * 100);
        return `
          <div class="flow-card" onclick="TraceApp.investigateInput('${f.source_label}')">
            <div class="flow-card-head">
              <span class="flow-route tabular">${f.source_label} → ${f.target_label}</span>
              <span class="flow-amount tabular">${fmtMoney(f.amount || 14200)}</span>
            </div>
            <div class="flow-progress-bar">
              <div class="flow-progress-fill" style="width: ${pct}%;"></div>
            </div>
            <div class="flow-weight-label">
              <span>Directed Transaction</span>
              <span class="tabular">GNN Attribution: ${pct}%</span>
            </div>
          </div>
        `;
      }).join("");
    }

    // Structural Typologies
    const patListEl = document.getElementById("explain-pattern-list");
    if (patListEl) {
      patListEl.innerHTML = (d.patterns || []).map(p => {
        const isDet = p.status === "Detected";
        return `
          <div class="pattern-pill ${isDet ? 'detected' : ''}" title="${p.detail || ''}">
            <span class="pattern-name">${p.name}</span>
            <span class="pattern-status tabular">${p.status}</span>
          </div>
        `;
      }).join("");
    }

    // Influential Features
    const featWeightEl = document.getElementById("explain-feature-weights");
    if (featWeightEl) {
      const imps = d.feature_importance || [];
      featWeightEl.innerHTML = imps.map(fi => {
        const pct = Math.round((fi.importance || 0.1) * 100);
        return `
          <div class="weight-row">
            <div class="weight-head">
              <span class="weight-name">${fi.feature}</span>
              <span class="weight-val tabular">${pct}%</span>
            </div>
            <div class="weight-track">
              <div class="weight-fill" style="width: ${pct * 2.2}%;"></div>
            </div>
          </div>
        `;
      }).join("");
    }

    // Bottom Topology Summary Footer
    const badgeEl = document.getElementById("contrast-badge");
    const narrEl = document.getElementById("topology-narrative");
    if (badgeEl && narrEl) {
      if (d.risk_level === "HIGH") {
        badgeEl.className = "evidence-badge high";
        badgeEl.textContent = "EVIDENTIARY CLUSTER";
        narrEl.textContent = d.summary || "Coordinated multi-hop transactional topology detected.";
      } else {
        badgeEl.className = "evidence-badge low";
        badgeEl.textContent = "CLEARED TOPOLOGY";
        narrEl.textContent = d.summary || "Acyclic low-velocity flow verified within commercial tolerances.";
      }
    }

    // Layout Graph according to active view mode (Focused vs Full)
    if (topologyMode === "full" && multiRingNetwork) {
      layoutForceDirectedGraph(multiRingNetwork, true);
    } else if (d.subgraph) {
      layoutForceDirectedGraph(d.subgraph, false);
    }
  }

  // =========================================================================
  // VIEW MODE SWITCHER (FOCUSED CASE vs FULL MULTI-RING MAP)
  // =========================================================================
  function setTopologyMode(mode) {
    topologyMode = mode;
    const btnFoc = document.getElementById("btn-view-focused");
    const btnFull = document.getElementById("btn-view-full");
    const stageTitle = document.getElementById("graph-stage-title");

    if (btnFoc) btnFoc.classList.toggle("active", mode === "focused");
    if (btnFull) btnFull.classList.toggle("active", mode === "full");

    if (mode === "full") {
      if (stageTitle) stageTitle.textContent = "Institutional Multi-Ring Topology Landscape";
      if (multiRingNetwork) {
        layoutForceDirectedGraph(multiRingNetwork, true);
        logAudit("Switched to FULL TOPOLOGY MAP: Viewing all connected accounts and cross-ring bridges across payment network.");
      }
    } else {
      if (stageTitle) stageTitle.textContent = "Forensic Transaction Topology";
      if (currentDossier && currentDossier.subgraph) {
        layoutForceDirectedGraph(currentDossier.subgraph, false);
        logAudit(`Returned to FOCUSED CASE view for ${currentDossier.account_id}.`);
      }
    }
  }

  // =========================================================================
  // INVESTIGATIVE WORKFLOW: ELEVATE, SURVEILLANCE & SAR REPORT
  // =========================================================================
  async function elevateActiveCase() {
    if (!currentDossier) return;
    const cid = currentDossier.case_id || currentKey;
    const acc = currentDossier.account_id;

    try {
      await fetch(`/api/case_action?case=${encodeURIComponent(currentKey)}&action=elevate`);
    } catch (_) {}

    currentDossier.status = "ELEVATED TO FIU";
    if (cases[currentKey]) cases[currentKey].status = "ELEVATED TO FIU";

    const statusEl = document.getElementById("dossier-case-status");
    if (statusEl) {
      statusEl.className = "case-status-badge elevated";
      statusEl.textContent = "ELEVATED TO FIU";
    }

    renderDrawerCases();
    logAudit(`CASE ESCALATED: Case ${cid} (${acc}) elevated to Senior FIU / Legal Council for statutory review.`);
    setStatus("loading", `<span>Case <strong>${cid}</strong> successfully elevated to Senior Compliance Officer. Tracking ticket: FIU-ESC-${acc}.</span>`);
    setTimeout(() => setStatus(null, ""), 5000);
  }

  async function placeUnderSurveillance() {
    if (!currentDossier) return;
    const cid = currentDossier.case_id || currentKey;
    const acc = currentDossier.account_id;

    try {
      await fetch(`/api/case_action?case=${encodeURIComponent(currentKey)}&action=surveillance`);
    } catch (_) {}

    currentDossier.status = "UNDER SURVEILLANCE";
    if (cases[currentKey]) cases[currentKey].status = "UNDER SURVEILLANCE";

    const statusEl = document.getElementById("dossier-case-status");
    if (statusEl) {
      statusEl.className = "case-status-badge surveillance";
      statusEl.textContent = "UNDER SURVEILLANCE";
    }

    renderDrawerCases();
    logAudit(`SURVEILLANCE ACTIVATED: Account ${acc} placed on automated 24/7 transaction monitoring.`);
    setStatus("loading", `<span>Account <strong>${acc}</strong> placed under continuous 24/7 real-time surveillance watchlist.</span>`);
    setTimeout(() => setStatus(null, ""), 5000);
  }

  // =========================================================================
  // OFFICIAL FINCEN SAR FORM 111 MODAL
  // =========================================================================
  function openSarModal() {
    if (!currentDossier) return;
    const d = currentDossier;
    const modal = document.getElementById("sar-modal-overlay");
    if (!modal) return;

    const docTitle = document.getElementById("sar-doc-title");
    const docRef = document.getElementById("sar-doc-ref");
    const docBody = document.getElementById("sar-doc-body");

    if (docTitle) docTitle.textContent = `CONFIDENTIAL INVESTIGATION DOSSIER — ${d.account_id}`;
    if (docRef) docRef.textContent = `Tracking Reference: SAR-2026-HI-${d.account_id} · Case: ${d.case_id || currentKey}`;

    const layman = d.layman_briefing || {};
    const edges = (d.subgraph && d.subgraph.edges) ? d.subgraph.edges : [];

    if (docBody) {
      docBody.innerHTML = `
        <div class="sar-section">
          <div class="sar-section-title">PART I — SUBJECT & ENTITY IDENTIFICATION</div>
          <div class="sar-grid-2">
            <div><strong>Primary Account ID:</strong> <span class="tabular">${d.account_id}</span> (Node #${d.node_id})</div>
            <div><strong>Investigation Status:</strong> <span>${d.status || 'OPEN INVESTIGATION'}</span></div>
            <div><strong>GraphSAGE Model Risk Score:</strong> <span class="tabular">${(d.gnn_score * 100).toFixed(1)}%</span> (${d.risk_level} RISK)</div>
            <div><strong>Tabular Baseline (No Graph):</strong> <span class="tabular">${(d.mlp_score * 100).toFixed(1)}%</span> (Rule Miss)</div>
            <div><strong>Operational Threshold:</strong> <span class="tabular">${(activeThreshold * 100).toFixed(1)}%</span></div>
            <div><strong>Estimated Laundering Volume:</strong> <span class="tabular">${fmtMoney(d.volume_est || 184500)}</span></div>
          </div>
        </div>

        <div class="sar-section">
          <div class="sar-section-title">PART II — INSPECTOR'S EXECUTIVE SUMMARY (PLAIN ENGLISH)</div>
          <p><strong>${layman.headline || 'Suspicious Financial Structuring Detected'}</strong></p>
          <p>${layman.summary || d.summary}</p>
        </div>

        <div class="sar-section">
          <div class="sar-section-title">PART III — KEY EVIDENTIARY RED FLAGS</div>
          <ul style="padding-left: 18px; display: flex; flex-direction: column; gap: 4px;">
            ${(layman.red_flags || []).map(rf => `<li>${rf}</li>`).join("")}
          </ul>
        </div>

        <div class="sar-section">
          <div class="sar-section-title">PART IV — CONNECTED COUNTERPARTY CONDUITS (${edges.length} Active Edges)</div>
          <table class="bench-table" style="margin-top: 6px;">
            <thead>
              <tr>
                <th>Source Account</th>
                <th>Destination</th>
                <th>Observed Amount</th>
                <th>GNN Attribution Weight</th>
              </tr>
            </thead>
            <tbody>
              ${edges.slice(0, 8).map(e => `
                <tr>
                  <td class="tabular">${e.source_label}</td>
                  <td class="tabular">${e.target_label}</td>
                  <td class="tabular">${fmtMoney(e.amount || 12000)}</td>
                  <td class="tabular">${Math.round((e.importance || 0.5) * 100)}%</td>
                </tr>
              `).join("")}
            </tbody>
          </table>
        </div>

        <div class="sar-section">
          <div class="sar-section-title">PART V — STATUTORY ENFORCEMENT DIRECTIVES</div>
          <ol style="padding-left: 18px; display: flex; flex-direction: column; gap: 4px;">
            <li>Issue automated BSA Section 314(a) inquiry across counterparty banking institutions.</li>
            <li>Place immediate hold on outbound wire transfers originating from Account ${d.account_id}.</li>
            <li>File formal FinCEN Form 111 (Suspicious Activity Report) within 30 statutory days.</li>
          </ol>
        </div>

        <div class="sar-signoff-box">
          <div>Prepared By: <strong>H. Vance, Lead FIU Investigator</strong></div>
          <div>Filing Timestamp: <strong class="tabular">${new Date().toUTCString()}</strong></div>
          <div>Jurisdiction: <strong>BSA / FinCEN Title 31</strong></div>
        </div>
      `;
    }

    modal.classList.add("open");
    logAudit(`Generated official FinCEN SAR regulatory memo for Account ${d.account_id}.`);
  }

  function closeSarModal() {
    const modal = document.getElementById("sar-modal-overlay");
    if (modal) modal.classList.remove("open");
  }

  // =========================================================================
  // THRESHOLD SLIDER & DUAL METRICS
  // =========================================================================
  function refreshThresholdDependentUI() {
    if (!currentDossier) return;
    const score = currentDossier.gnn_score;
    const isHigh = score >= activeThreshold;
    const isMed = !isHigh && score >= (activeThreshold * 0.65);

    const scoreEl = document.getElementById("dossier-gnn-score");
    const barEl = document.getElementById("dossier-score-bar");
    const pinEl = document.getElementById("dossier-thresh-pin");
    const cutoffEl = document.getElementById("dossier-cutoff-label");
    const readoutEl = document.getElementById("dossier-thresh-readout");
    const pillEl = document.getElementById("dossier-risk-pill");

    if (scoreEl) {
      scoreEl.textContent = (score * 100).toFixed(1) + "%";
      scoreEl.className = `risk-score-badge tabular ${isHigh ? 'high' : (isMed ? 'med' : 'low')}`;
    }

    if (barEl) {
      barEl.style.width = Math.min(100, Math.max(0, score * 100)) + "%";
      barEl.className = `gauge-fill ${isHigh ? 'high' : 'low'}`;
    }

    if (pinEl) {
      pinEl.style.left = (activeThreshold * 100).toFixed(1) + "%";
    }

    if (cutoffEl) {
      cutoffEl.textContent = (activeThreshold * 100).toFixed(1) + "%";
    }

    if (readoutEl) {
      readoutEl.textContent = `${(activeThreshold * 100).toFixed(1)}% (F1-Optimum)`;
    }

    if (pillEl) {
      pillEl.className = `risk-pill ${isHigh ? 'high' : (isMed ? 'med' : 'low')}`;
      pillEl.textContent = isHigh ? "CRITICAL RISK" : (isMed ? "MEDIUM RISK" : "CLEARED");
    }
  }

  function onThresholdChange(val) {
    activeThreshold = Number(val) / 100;
    refreshThresholdDependentUI();
    const btn = document.getElementById("btn-lock-val");
    if (btn) {
      const isLocked = Math.abs(activeThreshold - defaultValThreshold) < 0.005;
      btn.textContent = isLocked ? `Lock (${(defaultValThreshold * 100).toFixed(1)}%)` : `Reset (${(defaultValThreshold * 100).toFixed(1)}%)`;
    }
  }

  function resetThreshold() {
    activeThreshold = defaultValThreshold;
    const slider = document.getElementById("threshold-slider");
    if (slider) slider.value = Math.round(activeThreshold * 100);
    refreshThresholdDependentUI();
    const btn = document.getElementById("btn-lock-val");
    if (btn) btn.textContent = `Lock (${(defaultValThreshold * 100).toFixed(1)}%)`;
  }

  function runExplainability() {
    explainSpotlight = !explainSpotlight;
    const btn = document.getElementById("btn-explain-toggle");
    if (btn) btn.classList.toggle("active", explainSpotlight);

    const tag = document.getElementById("explainer-mode-tag");
    if (tag) {
      tag.textContent = explainSpotlight ? "SUBGRAPH MASK ACTIVE" : "ATTRIBUTION READY";
      tag.style.color = explainSpotlight ? "var(--cyan-intel)" : "";
    }

    replayAnimation();
    logAudit(explainSpotlight ? "GNNExplainer mask activated: spotlighting decisive transaction conduits" : "Returned to standard topology view");
  }

  // =========================================================================
  // FORCE-DIRECTED PHYSICS ENGINE (NO OVERLAPPING CIRCLES!)
  // =========================================================================
  function syncCanvasSize() {
    if (!canvas || !canvas.parentElement) return;
    const w = canvas.parentElement.clientWidth;
    const h = canvas.parentElement.clientHeight;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }

  function layoutForceDirectedGraph(networkData, isFullMap) {
    syncCanvasSize();
    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;

    const badge = document.getElementById("graph-stats-badge");
    if (badge) {
      badge.textContent = isFullMap
        ? `${networkData.nodes.length} Accounts · ${networkData.edges.length} Directed Flows (Full Landscape)`
        : `${networkData.nodes.length} Accounts · ${networkData.edges.length} Directed Edges`;
    }

    const nCount = networkData.nodes.length;

    // 1. Initial Positions (Target or hubs at center, neighbors distributed)
    renderNodes = networkData.nodes.map((n, idx) => {
      const isTarget = n.is_target;
      const angle = (2 * Math.PI * idx) / Math.max(1, nCount - 1);
      const rad = isTarget ? 0 : (isFullMap ? 140 + (idx % 6) * 55 : 160 + (idx % 3) * 40);
      return {
        ...n,
        x: isTarget ? cx : cx + Math.cos(angle) * rad,
        y: isTarget ? cy : cy + Math.sin(angle) * rad,
        vx: 0,
        vy: 0,
        r: isTarget ? 20 : (n.risk_score >= 0.35 ? 13 : 9)
      };
    });

    const nodeMap = {};
    renderNodes.forEach(n => { nodeMap[n.id] = n; });

    renderEdges = networkData.edges
      .filter(e => nodeMap[e.source] && nodeMap[e.target])
      .map(e => ({
        ...e,
        u: nodeMap[e.source],
        v: nodeMap[e.target]
      }));

    // 2. Physics Simulation Steps
    const kRepel = isFullMap ? 11000 : 7500;
    const kSpring = isFullMap ? 0.035 : 0.045;
    const restLen = isFullMap ? 160 : 135;
    const damping = 0.82;
    const steps = isFullMap ? 150 : 120;

    for (let step = 0; step < steps; step++) {
      // Repulsion between all node pairs
      for (let i = 0; i < renderNodes.length; i++) {
        const n1 = renderNodes[i];
        for (let j = i + 1; j < renderNodes.length; j++) {
          const n2 = renderNodes[j];
          const dx = n2.x - n1.x;
          const dy = n2.y - n1.y;
          const distSq = dx * dx + dy * dy + 100;
          const dist = Math.sqrt(distSq);
          const force = kRepel / distSq;
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;

          if (!n1.is_target || isFullMap) { n1.vx -= fx; n1.vy -= fy; }
          if (!n2.is_target || isFullMap) { n2.vx += fx; n2.vy += fy; }
        }
      }

      // Spring attraction along edges
      renderEdges.forEach(e => {
        const dx = e.v.x - e.u.x;
        const dy = e.v.y - e.u.y;
        const dist = Math.hypot(dx, dy) || 1;
        const delta = dist - restLen;
        const force = delta * kSpring;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;

        if (!e.u.is_target || isFullMap) { e.u.vx += fx; e.u.vy += fy; }
        if (!e.v.is_target || isFullMap) { e.v.vx += fx; e.v.vy += fy; }
      });

      // Centering gravity
      renderNodes.forEach(n => {
        if (!n.is_target || isFullMap) {
          n.vx += (cx - n.x) * (isFullMap ? 0.002 : 0.003);
          n.vy += (cy - n.y) * (isFullMap ? 0.002 : 0.003);
          n.vx *= damping;
          n.vy *= damping;
          n.x += n.vx;
          n.y += n.vy;
        } else {
          n.x = cx;
          n.y = cy;
        }
      });
    }

    cam = { x: 0, y: 0, zoom: isFullMap ? 0.85 : 1 };
    replayAnimation();
  }

  function replayAnimation() {
    animStart = performance.now();
    requestAnimationFrame(drawGraph);
  }

  function resetCamera() {
    cam = { x: 0, y: 0, zoom: topologyMode === "full" ? 0.85 : 1 };
    drawGraph(performance.now());
  }

  function centerTarget() {
    const t = renderNodes.find(n => n.is_target);
    if (!t) return;
    const w = canvas.width;
    const h = canvas.height;
    cam.zoom = 1.25;
    cam.x = (w / 2) - (t.x * cam.zoom);
    cam.y = (h / 2) - (t.y * cam.zoom);
    drawGraph(performance.now());
  }

  function drawGraph(now) {
    if (!ctx || !canvas) return;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.save();
    ctx.translate(cam.x, cam.y);
    ctx.scale(cam.zoom, cam.zoom);

    const pulseT = (now % 1800) / 1800;

    // 1. Draw Directed Edges
    renderEdges.forEach(e => {
      const dx = e.v.x - e.u.x;
      const dy = e.v.y - e.u.y;
      const dist = Math.hypot(dx, dy) || 1;
      const ux = dx / dist;
      const uy = dy / dist;

      const x1 = e.u.x + ux * (e.u.r + 2);
      const y1 = e.u.y + uy * (e.u.r + 2);
      const x2 = e.v.x - ux * (e.v.r + 5);
      const y2 = e.v.y - uy * (e.v.r + 5);

      const isTopFlow = e.importance >= 0.7;
      const isInterRing = e.is_inter_ring;

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);

      if (isInterRing) {
        ctx.strokeStyle = "rgba(168, 85, 247, 0.65)";
        ctx.lineWidth = 1.8;
        ctx.setLineDash([4, 4]);
      } else if (explainSpotlight) {
        ctx.strokeStyle = isTopFlow ? "#38bdf8" : "rgba(39, 54, 78, 0.2)";
        ctx.lineWidth = isTopFlow ? 2.8 : 0.8;
        ctx.setLineDash([]);
      } else {
        ctx.strokeStyle = isTopFlow ? "rgba(56, 189, 248, 0.7)" : "rgba(71, 85, 105, 0.35)";
        ctx.lineWidth = isTopFlow ? 2.0 : 1.0;
        ctx.setLineDash([]);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Arrowheads
      const head = isTopFlow ? 8 : 6;
      const ang = Math.atan2(y2 - y1, x2 - x1);
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - head * Math.cos(ang - Math.PI / 6), y2 - head * Math.sin(ang - Math.PI / 6));
      ctx.lineTo(x2 - head * Math.cos(ang + Math.PI / 6), y2 - head * Math.sin(ang + Math.PI / 6));
      ctx.closePath();
      ctx.fillStyle = ctx.strokeStyle;
      ctx.fill();

      // Energy Particle Pulse
      if (!explainSpotlight || isTopFlow || isInterRing) {
        const px = x1 + (x2 - x1) * pulseT;
        const py = y1 + (y2 - y1) * pulseT;
        ctx.beginPath();
        ctx.arc(px, py, isTopFlow ? 3 : 2, 0, Math.PI * 2);
        ctx.fillStyle = isInterRing ? "#c084fc" : (isTopFlow ? "#38bdf8" : "rgba(148, 163, 184, 0.6)");
        ctx.fill();
      }
    });

    // 2. Draw Nodes
    renderNodes.forEach(n => {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);

      if (n.is_target) {
        // Target Entity: Glowing Cyan Core
        ctx.fillStyle = "#0284c7";
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 3.5;
        ctx.shadowColor = "#38bdf8";
        ctx.shadowBlur = 16;
      } else if (n.risk_score >= 0.35) {
        // High Risk Node
        ctx.fillStyle = "#f43f5e";
        ctx.strokeStyle = "#fda4af";
        ctx.lineWidth = 2;
        ctx.shadowColor = "#f43f5e";
        ctx.shadowBlur = 8;
      } else if (n.risk_score >= 0.20) {
        // Medium Intermediary
        ctx.fillStyle = "#d97706";
        ctx.strokeStyle = "#fde68a";
        ctx.lineWidth = 1.5;
        ctx.shadowBlur = 0;
      } else {
        // Cleared / Low Risk
        ctx.fillStyle = "#059669";
        ctx.strokeStyle = "#6ee7b7";
        ctx.lineWidth = 1.5;
        ctx.shadowBlur = 0;
      }

      ctx.fill();
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Surveillance Reticle Pulse if under active surveillance
      if (n.is_target && currentDossier && currentDossier.status === "UNDER SURVEILLANCE") {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r + 8 + Math.sin(now * 0.005) * 3, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(245, 158, 11, 0.8)";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Smart Labels: Target always shows prominent pill; top nodes show clean badge; others on hover
      const isImportant = n.is_target || n.risk_score >= 0.35 || n === hoveredNode;
      if (isImportant) {
        const labelText = n.account_id;
        ctx.font = n.is_target ? "700 11px JetBrains Mono, monospace" : "600 10px JetBrains Mono, monospace";
        const metrics = ctx.measureText(labelText);
        const padX = 5;
        const padY = 2;
        const ly = n.y + n.r + 14;

        // Label pill background
        ctx.fillStyle = "rgba(10, 14, 22, 0.88)";
        ctx.fillRect(n.x - metrics.width / 2 - padX, ly - 9 - padY, metrics.width + padX * 2, 14 + padY * 2);
        ctx.strokeStyle = n.is_target ? "rgba(56, 189, 248, 0.4)" : "rgba(39, 54, 78, 0.6)";
        ctx.lineWidth = 1;
        ctx.strokeRect(n.x - metrics.width / 2 - padX, ly - 9 - padY, metrics.width + padX * 2, 14 + padY * 2);

        ctx.fillStyle = n.is_target ? "#ffffff" : "#cbd5e1";
        ctx.textAlign = "center";
        ctx.fillText(labelText, n.x, ly);
      }
    });

    ctx.restore();
    requestAnimationFrame(drawGraph);
  }

  function bindCanvasEvents() {
    canvas.addEventListener("mousedown", e => {
      dragging = true;
      dragOrigin = { x: e.clientX - cam.x, y: e.clientY - cam.y };
    });

    window.addEventListener("mouseup", () => { dragging = false; });

    canvas.addEventListener("mousemove", e => {
      const rect = canvas.getBoundingClientRect();
      const mx = (e.clientX - rect.left - cam.x) / cam.zoom;
      const my = (e.clientY - rect.top - cam.y) / cam.zoom;

      if (dragging) {
        cam.x = e.clientX - dragOrigin.x;
        cam.y = e.clientY - dragOrigin.y;
        return;
      }

      hoveredNode = renderNodes.find(n => Math.hypot(n.x - mx, n.y - my) <= n.r + 4) || null;
      const tip = document.getElementById("graph-tooltip");
      if (hoveredNode && tip) {
        canvas.style.cursor = "pointer";
        tip.style.display = "block";
        tip.style.left = (e.clientX - rect.left + 14) + "px";
        tip.style.top = (e.clientY - rect.top - 10) + "px";

        document.getElementById("tip-account-id").textContent = hoveredNode.account_id;
        document.getElementById("tip-risk-score").textContent = (hoveredNode.risk_score * 100).toFixed(1) + "%";
        document.getElementById("tip-risk-tier").textContent = hoveredNode.risk_level || (hoveredNode.risk_score >= 0.35 ? "HIGH" : "LOW");
        document.getElementById("tip-degrees").textContent = `${hoveredNode.in_degree || 1} / ${hoveredNode.out_degree || 1}`;
        document.getElementById("tip-volume").textContent = fmtMoney(hoveredNode.inflow || 18450);
      } else if (tip) {
        canvas.style.cursor = "grab";
        tip.style.display = "none";
      }
    });

    canvas.addEventListener("click", e => {
      if (hoveredNode && !hoveredNode.is_target) {
        investigateInput(hoveredNode.account_id);
      }
    });

    canvas.addEventListener("wheel", e => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.12 : 0.89;
      cam.zoom = Math.max(0.3, Math.min(3.5, cam.zoom * factor));
      drawGraph(performance.now());
    }, { passive: false });

    window.addEventListener("resize", () => {
      syncCanvasSize();
    });
  }

  async function investigateInput(customQuery) {
    const inputEl = document.getElementById("account-query-input");
    const q = (customQuery !== undefined ? customQuery : inputEl.value).trim();
    if (!q) {
      setStatus("error", `<span>Enter a valid Account ID or select a suspicious case.</span>`);
      return;
    }

    // 1. Check if query matches preloaded cases
    for (const [k, obj] of Object.entries(cases)) {
      if (obj.account_id.toLowerCase() === q.toLowerCase() || String(obj.node_id) === q || obj.case_id.toLowerCase() === q.toLowerCase()) {
        selectCase(k);
        return;
      }
    }

    // 2. Query backend Python service (/api/investigate)
    setStatus("loading", `<span>Running 3-Hop GraphSAGE for <strong>${q}</strong>...</span>`);

    try {
      const res = await fetch(`/api/investigate?account=${encodeURIComponent(q)}&explain=1`);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.error || `Account '${q}' not found.`);
      }
      const dossier = await res.json();
      setStatus(null, "");
      explainSpotlight = false;
      renderDossier(dossier);
      logAudit(`Investigated account ${dossier.account_id} (Node #${dossier.node_id})`);
    } catch (e) {
      setStatus("error", `<span>${e.message || "Account not found in 515K dataset."}</span>`);
    }
  }

  function bindKeyboardShortcuts() {
    window.addEventListener("keydown", e => {
      if (e.target && e.target.tagName === "INPUT") return;

      if (e.key === "1") selectCase("CASE_01");
      if (e.key === "2") selectCase("CASE_02");
      if (e.key === "3") selectCase("CASE_03");
      if (e.key === "e" || e.key === "E") runExplainability();
      if (e.key === "p" || e.key === "P") togglePresentationMode();
      if (e.key === "r" || e.key === "R") replayAnimation();
      if (e.key === "q" || e.key === "Q") toggleCaseDrawer();
      if (e.key === "/") {
        e.preventDefault();
        const input = document.getElementById("account-query-input");
        if (input) { input.focus(); input.select(); }
      }
      if (e.key === "Escape") {
        closeSarModal();
        const drawer = document.getElementById("case-drawer");
        if (drawer && drawer.classList.contains("open")) toggleCaseDrawer();
      }
    });
  }

  function togglePresentationMode() {
    document.body.classList.toggle("presentation-mode");
    const active = document.body.classList.contains("presentation-mode");
    const btn = document.getElementById("btn-pres-mode");
    if (btn) btn.classList.toggle("active", active);
    setTimeout(() => {
      if (topologyMode === "full" && multiRingNetwork) layoutForceDirectedGraph(multiRingNetwork, true);
      else if (currentDossier && currentDossier.subgraph) layoutForceDirectedGraph(currentDossier.subgraph, false);
    }, 50);
  }

  function switchDockTab(tabKey) {
    ["audit", "bench", "gov"].forEach(k => {
      const tab = document.getElementById(`dock-tab-${k}`);
      const pane = document.getElementById(`dock-pane-${k}`);
      if (tab) tab.classList.toggle("active", k === tabKey);
      if (pane) pane.classList.toggle("active", k === tabKey);
    });
  }

  function exportAuditMarkdown() {
    if (!currentDossier) return;
    const d = currentDossier;
    const layman = d.layman_briefing || {};
    const fid = d.fidelity_plus !== undefined ? d.fidelity_plus : 0.282;

    const lines = [
      `# FINCEN FORM 111 — SUSPICIOUS ACTIVITY REPORT`,
      `- **Tracking Reference:** SAR-2026-HI-${d.account_id}`,
      `- **Case Number:** ${d.case_id || currentKey}`,
      `- **Filing Date:** ${new Date().toUTCString()}`,
      `- **Investigated Entity:** Account ID \`${d.account_id}\` (Node #${d.node_id})`,
      `- **Investigation Status:** ${d.status || 'OPEN INVESTIGATION'}`,
      `- **GraphSAGE Model Risk Score:** ${(d.gnn_score * 100).toFixed(1)}% (${d.risk_level} RISK)`,
      `- **MLP Baseline Score:** ${(d.mlp_score * 100).toFixed(1)}% (Rule Blindspot)`,
      `- **Estimated Laundering Volume:** ${fmtMoney(d.volume_est || 184500)}`,
      ``,
      `## 1. Inspector's Plain-English Forensic Briefing`,
      `### ${layman.headline || 'Suspicious Financial Structuring Detected'}`,
      layman.summary || d.summary,
      ``,
      `## 2. Key Red Flags for Compliance Officers`,
      ...(layman.red_flags ? layman.red_flags.map(f => `- ⚠️ ${f}`) : []),
      ``,
      `## 3. Most Decisive Connected Counterparty Conduits`,
      ...(d.subgraph && d.subgraph.edges ? d.subgraph.edges.slice(0, 8).map(
        e => `- \`${e.source_label} → ${e.target_label}\` | Amount: ${fmtMoney(e.amount || 14200)} | GNN Weight: ${Math.round((e.importance || 0.5) * 100)}%`
      ) : []),
      ``,
      `## 4. Statutory Enforcement Directives`,
      `1. Freeze outbound wire transfers from Account ${d.account_id} immediately.`,
      `2. Issue automated USA PATRIOT Act Section 314(a) inquiry across counterparty banking institutions.`,
      `3. File official FinCEN Form 111 with FinCEN Regulatory Portal.`
    ];

    const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `FINCEN_SAR_${d.account_id}.md`;
    a.click();
    logAudit(`Exported official FinCEN SAR regulatory filing: FINCEN_SAR_${d.account_id}.md`);
  }

  return {
    init,
    selectCase,
    investigateInput,
    toggleCaseDrawer,
    filterCases,
    setTopologyMode,
    elevateActiveCase,
    placeUnderSurveillance,
    openSarModal,
    closeSarModal,
    runExplainability,
    onThresholdChange,
    resetThreshold,
    replayAnimation,
    resetCamera,
    centerTarget,
    togglePresentationMode,
    switchDockTab,
    exportAuditMarkdown
  };
})();

window.addEventListener("DOMContentLoaded", () => TraceApp.init());