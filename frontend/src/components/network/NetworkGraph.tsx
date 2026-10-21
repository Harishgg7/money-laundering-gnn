import React, { useRef, useEffect, useState, useCallback } from 'react';
import { NetworkGraphData, NetworkNode, NetworkEdge } from '../../types';
import { computeTargetCenteredLayout, LayoutResult } from './networkLayout';

interface NetworkGraphProps {
  data: NetworkGraphData;
  selectedNodeId: string | null;
  onSelectNode: (node: NetworkNode | null) => void;
  targetAccountId?: string | null;
  height?: number | string;
}

export const NetworkGraph: React.FC<NetworkGraphProps> = ({
  data,
  selectedNodeId,
  onSelectNode,
  targetAccountId,
  height = '100%',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Viewport Camera State
  const [camera, setCamera] = useState({ x: 0, y: 0, zoom: 1 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [hoveredNode, setHoveredNode] = useState<(NetworkNode & { hopLevel?: number }) | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });

  // Positioned layout references
  const layoutRef = useRef<LayoutResult | null>(null);
  const nodesRef = useRef<(NetworkNode & { x: number; y: number; hopLevel: number })[]>([]);
  const edgesRef = useRef<NetworkEdge[]>([]);

  // Recenter whole graph to fit current nodes neatly with target at the exact center
  const recenter = useCallback(() => {
    if (!layoutRef.current || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const bounds = layoutRef.current.bounds;

    const availableWidth = rect.width * 0.84;
    const availableHeight = rect.height * 0.84;
    const scaleX = availableWidth / bounds.width;
    const scaleY = availableHeight / bounds.height;

    // For sparse graphs, avoid extreme zoom; for larger bounded graphs, scale down cleanly
    const rawFit = Math.min(scaleX, scaleY);
    const fitZoom = bounds.width < 220 && bounds.height < 220
      ? 1.0
      : Math.min(Math.max(rawFit, 0.35), 1.15);

    // Because target node is placed at (0, 0), placing camera at (rect.width / 2, rect.height / 2)
    // centers the target node and the whole concentric topology exactly in the viewport
    setCamera({
      x: rect.width / 2,
      y: rect.height / 2,
      zoom: fitZoom,
    });
  }, []);

  // Compute deterministic target-centered layout whenever data or target account changes
  useEffect(() => {
    const activeTarget = targetAccountId || null;
    const layout = computeTargetCenteredLayout(data.nodes, data.edges, activeTarget);
    layoutRef.current = layout;
    nodesRef.current = layout.nodes;
    edgesRef.current = layout.edges;

    // Auto-fit to viewport with target at visual center
    recenter();
  }, [data, targetAccountId, recenter]);

  // Center on a specific node at investigation zoom
  const centerOnNode = useCallback((node: NetworkNode & { x: number; y: number }, targetZoom = 1.15) => {
    if (!containerRef.current || node.x === undefined || node.y === undefined) return;
    const rect = containerRef.current.getBoundingClientRect();
    setCamera({
      x: rect.width / 2 - node.x * targetZoom,
      y: rect.height / 2 - node.y * targetZoom,
      zoom: targetZoom,
    });
  }, []);

  // Crisp HiDPI Canvas Rendering Loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animId: number;

    const render = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;

      if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
      }

      ctx.save();
      ctx.scale(dpr, dpr);

      // Clean Light Mode Canvas Background
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, rect.width, rect.height);

      // Background subtle dot matrix grid
      const gridSize = 28 * camera.zoom;
      if (camera.zoom >= 0.55) {
        ctx.fillStyle = '#f1f5f9';
        const startX = ((camera.x % gridSize) + gridSize) % gridSize;
        const startY = ((camera.y % gridSize) + gridSize) % gridSize;
        for (let x = startX - gridSize; x < rect.width + gridSize; x += gridSize) {
          for (let y = startY - gridSize; y < rect.height + gridSize; y += gridSize) {
            ctx.fillRect(x, y, 1.5, 1.5);
          }
        }
      }

      ctx.translate(camera.x, camera.y);
      ctx.scale(camera.zoom, camera.zoom);

      // Map nodes by ID for edge lookup
      const nodeMap = new Map<string, NetworkNode & { x: number; y: number; hopLevel: number }>();
      nodesRef.current.forEach((n) => nodeMap.set(n.id, n));

      // Resolve currently selected canonical node and its direct neighbors
      const activeSelectedId = selectedNodeId || targetAccountId || null;
      let selectedNodeObj: (NetworkNode & { x: number; y: number }) | undefined;
      if (activeSelectedId) {
        selectedNodeObj = nodesRef.current.find(
          (n) => n.id === activeSelectedId || n.accountId === activeSelectedId
        );
      }
      const activeId = selectedNodeObj ? selectedNodeObj.id : activeSelectedId;

      const neighborSet = new Set<string>();
      if (activeId) {
        neighborSet.add(activeId);
        edgesRef.current.forEach((e) => {
          if (e.source === activeId) neighborSet.add(e.target);
          if (e.target === activeId) neighborSet.add(e.source);
        });
      }
      const hasSelection = neighborSet.size > 0;

      // 0. Draw Subtle Concentric Guide Rings around Target (0, 0)
      const has1Hop = nodesRef.current.some((n) => n.hopLevel === 1);
      const has2Hop = nodesRef.current.some((n) => n.hopLevel === 2);

      if (has1Hop || has2Hop) {
        ctx.save();
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 6]);

        // Find average radius of 1-hop and 2-hop rings for clean guide circles
        let r1Sum = 0, r1Count = 0;
        let r2Sum = 0, r2Count = 0;
        nodesRef.current.forEach((n) => {
          const r = Math.sqrt(n.x * n.x + n.y * n.y);
          if (n.hopLevel === 1) { r1Sum += r; r1Count++; }
          else if (n.hopLevel === 2) { r2Sum += r; r2Count++; }
        });

        if (r1Count > 0) {
          const r1Avg = r1Sum / r1Count;
          ctx.beginPath();
          ctx.arc(0, 0, r1Avg, 0, Math.PI * 2);
          ctx.stroke();

          // Concentric Layer Badge
          ctx.save();
          ctx.setLineDash([]);
          ctx.font = '600 9.5px Inter, sans-serif';
          ctx.fillStyle = '#94a3b8';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText('1-HOP DIRECT', r1Avg + 8, -6);
          ctx.restore();
        }

        if (r2Count > 0) {
          const r2Avg = r2Sum / r2Count;
          ctx.beginPath();
          ctx.arc(0, 0, r2Avg, 0, Math.PI * 2);
          ctx.stroke();

          ctx.save();
          ctx.setLineDash([]);
          ctx.font = '600 9.5px Inter, sans-serif';
          ctx.fillStyle = '#94a3b8';
          ctx.textAlign = 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText('2-HOP SECONDARY', r2Avg + 8, -6);
          ctx.restore();
        }
        ctx.restore();
      }

      // 1. Draw Edges with Strict Hierarchical Weighting
      edgesRef.current.forEach((e) => {
        const src = nodeMap.get(e.source);
        const dst = nodeMap.get(e.target);
        if (!src || !dst || src.x === undefined || src.y === undefined || dst.x === undefined || dst.y === undefined) return;

        const isDirectSubjectEdge = src.isTarget || dst.isTarget;
        const isConnectedToSelected = activeId === src.id || activeId === dst.id;
        const isNeighborEdge = neighborSet.has(src.id) && neighborSet.has(dst.id);

        const dx = dst.x - src.x;
        const dy = dst.y - src.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist === 0) return;

        // Base width proportional to transaction volume
        const baseWidth = Math.min(Math.max((e.amount / 180000) * 1.5, 1.2), 3.6);

        ctx.beginPath();
        ctx.moveTo(src.x, src.y);
        ctx.lineTo(dst.x, dst.y);

        if (isConnectedToSelected) {
          // Strongly emphasized path for active selection
          ctx.strokeStyle = '#2563eb';
          ctx.lineWidth = baseWidth * 1.4;
          ctx.globalAlpha = 1.0;
        } else if (hasSelection) {
          if (isDirectSubjectEdge) {
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = baseWidth;
            ctx.globalAlpha = 0.65;
          } else if (isNeighborEdge) {
            ctx.strokeStyle = '#94a3b8';
            ctx.lineWidth = Math.min(baseWidth, 1.6);
            ctx.globalAlpha = 0.45;
          } else {
            // Secondary background relationships dim significantly
            ctx.strokeStyle = '#cbd5e1';
            ctx.lineWidth = 1;
            ctx.globalAlpha = 0.12;
          }
        } else {
          // Default Hierarchical Edge Visualization:
          if (isDirectSubjectEdge) {
            // Direct 1-hop links to subject stand out clearly
            ctx.strokeStyle = '#3b82f6';
            ctx.lineWidth = Math.max(baseWidth, 1.6);
            ctx.globalAlpha = 0.82;
          } else {
            // 2-hop secondary links remain subdued
            ctx.strokeStyle = '#94a3b8';
            ctx.lineWidth = Math.min(baseWidth, 1.4);
            ctx.globalAlpha = 0.38;
          }
        }
        ctx.stroke();

        // Directional Arrow Head
        const dstRadius = dst.isTarget ? 20 : dst.hopLevel === 1 ? 16 : 14;
        const arrowDist = dist - dstRadius - 4;
        if (arrowDist > 10) {
          const arrowX = src.x + (dx / dist) * arrowDist;
          const arrowY = src.y + (dy / dist) * arrowDist;
          const angle = Math.atan2(dy, dx);
          const arrowSize = isConnectedToSelected ? 6.5 : 5.0;

          ctx.save();
          ctx.translate(arrowX, arrowY);
          ctx.rotate(angle);
          ctx.beginPath();
          ctx.moveTo(0, 0);
          ctx.lineTo(-arrowSize * 1.4, -arrowSize * 0.85);
          ctx.lineTo(-arrowSize * 1.4, arrowSize * 0.85);
          ctx.closePath();
          ctx.fillStyle = isConnectedToSelected
            ? '#2563eb'
            : isDirectSubjectEdge
            ? '#3b82f6'
            : hasSelection
            ? isNeighborEdge
              ? '#94a3b8'
              : '#cbd5e1'
            : '#94a3b8';
          ctx.fill();
          ctx.restore();
        }

        ctx.globalAlpha = 1.0;
      });

      // 2. Draw Nodes with Clear Visual Hierarchy & Radially Oriented Labels
      nodesRef.current.forEach((n) => {
        if (n.x === undefined || n.y === undefined) return;

        const isSelected = activeId === n.id || activeId === n.accountId;
        const isNeighbor = neighborSet.has(n.id);
        const isTarget = n.isTarget || targetAccountId === n.accountId;

        // Radius scaled by hop and volume
        let radius = 14;
        if (isTarget) {
          radius = 21;
        } else if (n.hopLevel === 1) {
          radius = Math.min(Math.max(14 + Math.log10(Math.max(n.totalVolume, 1000)) * 1.1, 15), 18);
        } else {
          radius = Math.min(Math.max(12 + Math.log10(Math.max(n.totalVolume, 1000)) * 0.7, 13), 15);
        }

        // De-emphasis logic for unrelated nodes
        const isDimmed = hasSelection && !isNeighbor;
        ctx.globalAlpha = isDimmed ? 0.22 : 1.0;

        // Target / Selection Ring
        if (isTarget || isSelected) {
          ctx.beginPath();
          ctx.arc(n.x, n.y, radius + 7, 0, Math.PI * 2);
          ctx.fillStyle = 'rgba(37, 99, 235, 0.14)';
          ctx.fill();
          ctx.lineWidth = 2.5;
          ctx.strokeStyle = '#2563eb';
          ctx.stroke();

          // Subtle dashed outer focus halo for canonical target
          if (isTarget) {
            ctx.save();
            ctx.beginPath();
            ctx.arc(n.x, n.y, radius + 11, 0, Math.PI * 2);
            ctx.setLineDash([3, 4]);
            ctx.strokeStyle = 'rgba(37, 99, 235, 0.45)';
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.restore();
          }
        } else if (hasSelection && isNeighbor) {
          // Connected neighbor highlight ring
          ctx.beginPath();
          ctx.arc(n.x, n.y, radius + 3, 0, Math.PI * 2);
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = 'rgba(37, 99, 235, 0.45)';
          ctx.stroke();
        }

        // Node Circle Body
        ctx.beginPath();
        ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);

        // Risk Colors: High Red, Medium Amber, Low Green
        let nodeColor = '#16a34a'; // Low
        if (n.riskLevel === 'HIGH') nodeColor = '#dc2626'; // High
        else if (n.riskLevel === 'MEDIUM') nodeColor = '#d97706'; // Med

        ctx.fillStyle = nodeColor;
        ctx.fill();
        ctx.lineWidth = isSelected || isTarget ? 2.5 : 1.5;
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();

        // Node Account Label — Radially Oriented (Outward)
        // Upper hemisphere nodes (y < -35) place label ABOVE node so it never points inward
        // Lower hemisphere nodes place label BELOW node
        ctx.font = isTarget
          ? '700 11.5px Inter, sans-serif'
          : isSelected
          ? '700 11px Inter, sans-serif'
          : '500 10px Inter, sans-serif';
        ctx.textAlign = 'center';

        const label = n.accountId;
        const textWidth = ctx.measureText(label).width;
        const isUpper = !isTarget && n.y < -35;
        const textY = isUpper ? n.y - radius - 5 : n.y + radius + 5;
        ctx.textBaseline = isUpper ? 'bottom' : 'top';

        // Crisp rounded pill background for readability
        const pillY = isUpper ? textY - 14 : textY - 1;
        ctx.fillStyle = isDimmed ? 'rgba(255, 255, 255, 0.70)' : 'rgba(255, 255, 255, 0.95)';
        ctx.fillRect(n.x - textWidth / 2 - 5, pillY, textWidth + 10, 16);
        ctx.strokeStyle = isDimmed ? 'rgba(226, 232, 240, 0.45)' : '#e2e8f0';
        ctx.lineWidth = 1;
        ctx.strokeRect(n.x - textWidth / 2 - 5, pillY, textWidth + 10, 16);

        ctx.fillStyle = isTarget
          ? '#1e3a8a'
          : isSelected
          ? '#2563eb'
          : isDimmed
          ? '#94a3b8'
          : '#0f172a';
        ctx.fillText(label, n.x, textY);

        ctx.globalAlpha = 1.0;
      });

      ctx.restore();
      animId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animId);
  }, [camera, selectedNodeId, hoveredNode, targetAccountId]);

  // Screen Coordinates to Canvas Graph Coordinates
  const toGraphCoord = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const sx = clientX - rect.left;
    const sy = clientY - rect.top;
    return {
      x: (sx - camera.x) / camera.zoom,
      y: (sy - camera.y) / camera.zoom,
    };
  };

  // Find node at coordinates
  const findNodeAt = (graphX: number, graphY: number) => {
    for (let i = nodesRef.current.length - 1; i >= 0; i--) {
      const n = nodesRef.current[i];
      if (n.x === undefined || n.y === undefined) continue;
      const dx = graphX - n.x;
      const dy = graphY - n.y;
      const radius = n.isTarget ? 24 : 20;
      if (dx * dx + dy * dy <= radius * radius) {
        return n;
      }
    }
    return null;
  };

  // Mouse Interaction Handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    const { x, y } = toGraphCoord(e.clientX, e.clientY);
    const clickedNode = findNodeAt(x, y);

    if (clickedNode) {
      onSelectNode(clickedNode);
      centerOnNode(clickedNode, 1.15);
    } else {
      setIsDragging(true);
      setDragStart({ x: e.clientX - camera.x, y: e.clientY - camera.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    setMousePos({ x: e.clientX, y: e.clientY });

    if (isDragging) {
      setCamera((prev) => ({
        ...prev,
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      }));
    } else {
      const { x, y } = toGraphCoord(e.clientX, e.clientY);
      const node = findNodeAt(x, y);
      setHoveredNode(node);
      if (canvasRef.current) {
        canvasRef.current.style.cursor = node ? 'pointer' : 'default';
      }
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onWheelNative = (e: WheelEvent) => {
      e.preventDefault();
      const zoomFactor = e.deltaY < 0 ? 1.12 : 0.89;

      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      setCamera((prev) => {
        const newZoom = Math.min(Math.max(prev.zoom * zoomFactor, 0.35), 2.8);
        return {
          zoom: newZoom,
          x: mouseX - (mouseX - prev.x) * (newZoom / prev.zoom),
          y: mouseY - (mouseY - prev.y) * (newZoom / prev.zoom),
        };
      });
    };

    canvas.addEventListener('wheel', onWheelNative, { passive: false });
    return () => {
      canvas.removeEventListener('wheel', onWheelNative);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        width: '100%',
        height,
        backgroundColor: '#ffffff',
        border: '1px solid var(--border-subtle)',
        borderRadius: 'var(--radius-sm)',
        overflow: 'hidden',
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '100%', display: 'block' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />

      {/* Floating Viewport Controls */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          right: 12,
          display: 'flex',
          flexDirection: 'column',
          gap: 4,
          zIndex: 10,
        }}
      >
        <button
          className="btn btn-secondary btn-sm"
          style={{ width: 28, height: 28, padding: 0 }}
          onClick={() => setCamera((c) => ({ ...c, zoom: Math.min(c.zoom * 1.25, 2.8) }))}
          title="Zoom In"
        >
          +
        </button>
        <button
          className="btn btn-secondary btn-sm"
          style={{ width: 28, height: 28, padding: 0 }}
          onClick={() => setCamera((c) => ({ ...c, zoom: Math.max(c.zoom * 0.8, 0.35) }))}
          title="Zoom Out"
        >
          &minus;
        </button>
        <button
          className="btn btn-secondary btn-sm"
          style={{ width: 28, height: 28, padding: 0, fontSize: 10 }}
          onClick={recenter}
          title="Recenter Target"
        >
          &#x21bb;
        </button>
      </div>

      {/* Interactive Hover Tooltip */}
      {hoveredNode && (
        <div
          style={{
            position: 'fixed',
            left: Math.min(mousePos.x + 14, window.innerWidth - 270),
            top: Math.min(mousePos.y + 14, window.innerHeight - 170),
            backgroundColor: '#ffffff',
            border: '1px solid var(--border-default)',
            borderRadius: 'var(--radius-sm)',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)',
            padding: '10px 14px',
            fontSize: 11,
            zIndex: 40,
            pointerEvents: 'none',
            minWidth: 210,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8, marginBottom: 6 }}>
            <span className="mono" style={{ fontWeight: 700, fontSize: 13, color: 'var(--text-primary)' }}>
              {hoveredNode.accountId}
            </span>
            <span
              style={{
                fontSize: 10,
                fontWeight: 600,
                padding: '2px 6px',
                borderRadius: 4,
                backgroundColor:
                  hoveredNode.riskLevel === 'HIGH' ? '#fee2e2' : hoveredNode.riskLevel === 'MEDIUM' ? '#fef3c7' : '#dcfce7',
                color:
                  hoveredNode.riskLevel === 'HIGH' ? '#991b1b' : hoveredNode.riskLevel === 'MEDIUM' ? '#92400e' : '#166534',
              }}
            >
              {hoveredNode.riskLevel}
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '4px 10px', color: 'var(--text-secondary)' }}>
            <span style={{ color: 'var(--text-muted)' }}>Relationship:</span>
            <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
              {hoveredNode.isTarget ? 'Subject Account' : hoveredNode.hopLevel === 1 ? 'Direct Counterparty (1-Hop)' : 'Secondary Counterparty (2-Hop)'}
            </span>

            <span style={{ color: 'var(--text-muted)' }}>Model Risk Score:</span>
            <span className="mono tabular" style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
              {(hoveredNode.riskScore * 100).toFixed(2)}%
            </span>

            <span style={{ color: 'var(--text-muted)' }}>Total Volume:</span>
            <span className="mono tabular">
              ${hoveredNode.totalVolume.toLocaleString()}
            </span>

            <span style={{ color: 'var(--text-muted)' }}>Transactions:</span>
            <span className="tabular">{hoveredNode.transactionCount}</span>
          </div>
        </div>
      )}

      {/* Restrained Institutional Legend */}
      <div
        style={{
          position: 'absolute',
          bottom: 12,
          left: 12,
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-xs)',
          padding: '6px 12px',
          display: 'flex',
          alignItems: 'center',
          gap: 14,
          fontSize: 11,
          color: 'var(--text-secondary)',
          boxShadow: 'var(--shadow-xs)',
          pointerEvents: 'none',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: '#dc2626' }} />
          <span>High Risk (&ge;0.38)</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: '#d97706' }} />
          <span>Medium Risk</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
          <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: '#16a34a' }} />
          <span>Low Risk</span>
        </div>
        <div style={{ width: 1, height: 12, backgroundColor: 'var(--border-subtle)' }} />
        <div style={{ display: 'flex', alignItems: 'center', gap: 5, color: 'var(--text-muted)' }}>
          <span style={{ display: 'inline-block', width: 14, borderBottom: '1px dashed #94a3b8' }} />
          <span>Concentric Layers (1-Hop Direct, 2-Hop Secondary)</span>
        </div>
      </div>
    </div>
  );
};
