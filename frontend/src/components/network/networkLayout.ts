import { NetworkNode, NetworkEdge } from '../../types';

export interface LayoutResult {
  nodes: (NetworkNode & { x: number; y: number; hopLevel: number })[];
  edges: NetworkEdge[];
  bounds: {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
    width: number;
    height: number;
  };
}

/**
 * Computes an investigation-oriented, deterministic, target-centered radial/hierarchical layout.
 *
 * Rules:
 * 1. Target node is permanently positioned at (0, 0).
 * 2. 1-hop direct counterparties form an inner ring with even angular distribution.
 * 3. 2-hop secondary counterparties form an outer ring, clustered near their 1-hop parents.
 * 4. Deterministic: No Math.random(). Same graph = identical coordinates.
 * 5. Adaptive: Dynamically adjusts radii and staggers rings based on node count to prevent overlap.
 * 6. Minimum collision spacing: Guarantees adequate whitespace for labels and directional arrows.
 */
export function computeTargetCenteredLayout(
  nodes: NetworkNode[],
  edges: NetworkEdge[],
  targetAccountId?: string | null
): LayoutResult {
  if (!nodes || nodes.length === 0) {
    return {
      nodes: [],
      edges: [],
      bounds: { minX: 0, maxX: 0, minY: 0, maxY: 0, width: 0, height: 0 },
    };
  }

  // 1. Identify Canonical Target Node
  let targetNode = nodes.find(
    (n) =>
      (targetAccountId && (n.accountId.toLowerCase() === targetAccountId.toLowerCase() || n.id.toLowerCase() === targetAccountId.toLowerCase())) ||
      n.isTarget
  );

  if (!targetNode) {
    // Fallback: node with highest degree or highest risk
    const degreeMap = new Map<string, number>();
    edges.forEach((e) => {
      degreeMap.set(e.source, (degreeMap.get(e.source) || 0) + 1);
      degreeMap.set(e.target, (degreeMap.get(e.target) || 0) + 1);
    });

    let bestScore = -1;
    nodes.forEach((n) => {
      const deg = degreeMap.get(n.id) || 0;
      const score = deg * 10 + n.riskScore;
      if (score > bestScore) {
        bestScore = score;
        targetNode = n;
      }
    });
  }

  const targetId = targetNode?.id || nodes[0].id;

  // 2. Classify into Hops (0: Target, 1: Direct Counterparties, 2: Secondary)
  const directCounterparties = new Set<string>();
  const parentOf2Hop = new Map<string, string>(); // 2-hop node ID -> primary 1-hop parent ID

  edges.forEach((e) => {
    if (e.source === targetId) {
      directCounterparties.add(e.target);
    } else if (e.target === targetId) {
      directCounterparties.add(e.source);
    }
  });

  // For 2-hop nodes, record connection to 1-hop parent
  edges.forEach((e) => {
    if (directCounterparties.has(e.source) && !directCounterparties.has(e.target) && e.target !== targetId) {
      if (!parentOf2Hop.has(e.target)) parentOf2Hop.set(e.target, e.source);
    } else if (directCounterparties.has(e.target) && !directCounterparties.has(e.source) && e.source !== targetId) {
      if (!parentOf2Hop.has(e.source)) parentOf2Hop.set(e.source, e.target);
    }
  });

  const oneHopNodes: NetworkNode[] = [];
  const twoHopNodes: NetworkNode[] = [];

  nodes.forEach((n) => {
    if (n.id === targetId) return;
    if (directCounterparties.has(n.id)) {
      oneHopNodes.push(n);
    } else {
      twoHopNodes.push(n);
    }
  });

  // 3. Deterministic Sorting
  // Sort 1-hop nodes by Risk Level (HIGH -> MEDIUM -> LOW), then totalVolume desc, then accountId
  const riskRank = (lvl: string) => (lvl === 'HIGH' ? 0 : lvl === 'MEDIUM' ? 1 : 2);
  oneHopNodes.sort((a, b) => {
    const diff = riskRank(a.riskLevel) - riskRank(b.riskLevel);
    if (diff !== 0) return diff;
    if (b.totalVolume !== a.totalVolume) return b.totalVolume - a.totalVolume;
    return a.accountId.localeCompare(b.accountId);
  });

  // 4. Adaptive Radii Calculation
  const N1 = oneHopNodes.length;
  const N2 = twoHopNodes.length;
  const totalNonTarget = N1 + N2;

  // Generous concentric moats: 1-hop and 2-hop rings have clear radial separation
  let baseR1 = 185;
  if (N1 <= 3) baseR1 = 175;
  else if (N1 <= 6) baseR1 = 205;
  else if (N1 <= 10) baseR1 = 240;
  else if (N1 <= 16) baseR1 = 270;
  else baseR1 = 300;

  // Outer ring radius: guaranteed at least 150px - 190px wider than inner ring
  let baseR2 = baseR1 + 160;
  if (N2 <= 4) baseR2 = baseR1 + 145;
  else if (N2 <= 10) baseR2 = baseR1 + 175;
  else if (N2 <= 18) baseR2 = baseR1 + 200;
  else baseR2 = baseR1 + 225;

  // 5. Position 1-Hop Nodes (Inner Ring)
  const positionedNodes: (NetworkNode & { x: number; y: number; hopLevel: number })[] = [];
  const oneHopAngles = new Map<string, number>();

  // Add target at (0, 0)
  positionedNodes.push({
    ...targetNode!,
    isTarget: true,
    x: 0,
    y: 0,
    hopLevel: 0,
  });

  for (let i = 0; i < N1; i++) {
    const node = oneHopNodes[i];
    // Start from top (-π/2) and distribute evenly clockwise
    const angle = -Math.PI / 2 + (i * 2 * Math.PI) / Math.max(N1, 1);
    oneHopAngles.set(node.id, angle);

    // Stagger radius slightly if many 1-hop nodes to prevent horizontal label overlap
    const r1 = N1 > 8 ? (i % 2 === 0 ? baseR1 - 14 : baseR1 + 16) : baseR1;
    const x = Math.round(r1 * Math.cos(angle));
    const y = Math.round(r1 * Math.sin(angle));

    positionedNodes.push({
      ...node,
      isTarget: false,
      x,
      y,
      hopLevel: 1,
    });
  }

  // 6. Position 2-Hop Nodes (Outer Ring)
  // Group 2-hop nodes by their 1-hop parent
  const parentToChildren = new Map<string, NetworkNode[]>();
  const orphan2Hop: NetworkNode[] = [];

  twoHopNodes.forEach((node) => {
    const parentId = parentOf2Hop.get(node.id);
    if (parentId && oneHopAngles.has(parentId)) {
      if (!parentToChildren.has(parentId)) parentToChildren.set(parentId, []);
      parentToChildren.get(parentId)!.push(node);
    } else {
      orphan2Hop.push(node);
    }
  });

  // Deterministically sort children within each parent
  parentToChildren.forEach((children) => {
    children.sort((a, b) => {
      const diff = riskRank(a.riskLevel) - riskRank(b.riskLevel);
      if (diff !== 0) return diff;
      return a.accountId.localeCompare(b.accountId);
    });
  });

  orphan2Hop.sort((a, b) => a.accountId.localeCompare(b.accountId));

  // Place clustered 2-hop nodes in angular sectors near their parent
  parentToChildren.forEach((children, parentId) => {
    const parentAngle = oneHopAngles.get(parentId)!;
    const k = children.length;
    // Angular span allocated for this parent's cluster
    const maxSpan = Math.min(((2 * Math.PI) / Math.max(N1, 2)) * 0.88, k * 0.30);
    const angleStep = k > 1 ? maxSpan / (k - 1) : 0;
    const startAngle = parentAngle - maxSpan / 2;

    children.forEach((child, j) => {
      const angle = k === 1 ? parentAngle : startAngle + j * angleStep;
      // Stagger radius for adjacent children to guarantee clearance
      const r2 = k > 2 ? (j % 2 === 0 ? baseR2 - 18 : baseR2 + 20) : baseR2;
      const x = Math.round(r2 * Math.cos(angle));
      const y = Math.round(r2 * Math.sin(angle));

      positionedNodes.push({
        ...child,
        isTarget: false,
        x,
        y,
        hopLevel: 2,
      });
    });
  });

  // Distribute orphan 2-hop nodes evenly around outer ring in unused sectors
  const orphanCount = orphan2Hop.length;
  if (orphanCount > 0) {
    const orphanStartAngle = Math.PI / 4;
    for (let i = 0; i < orphanCount; i++) {
      const angle = orphanStartAngle + (i * 2 * Math.PI) / orphanCount;
      const r2 = baseR2 + 15;
      const x = Math.round(r2 * Math.cos(angle));
      const y = Math.round(r2 * Math.sin(angle));

      positionedNodes.push({
        ...orphan2Hop[i],
        isTarget: false,
        x,
        y,
        hopLevel: 2,
      });
    }
  }

  // 7. Deterministic Label-Aware Collision Resolution Pass
  // Pill labels are ~84px wide by ~48px high. We use an elliptical distance metric
  // to ensure adequate horizontal and vertical whitespace.
  const labelWidth = 86;
  const labelHeight = 52;
  const iterations = 16;

  for (let it = 0; it < iterations; it++) {
    for (let i = 1; i < positionedNodes.length; i++) {
      for (let j = i + 1; j < positionedNodes.length; j++) {
        const u = positionedNodes[i];
        const v = positionedNodes[j];
        const dx = v.x - u.x;
        const dy = v.y - u.y;

        // Normalized elliptical distance
        const normX = dx / labelWidth;
        const normY = dy / labelHeight;
        const normDist = Math.sqrt(normX * normX + normY * normY);

        if (normDist < 1.0 && normDist > 0.001) {
          const overlap = (1.0 - normDist) / normDist;
          const shiftX = dx * overlap * 0.28;
          const shiftY = dy * overlap * 0.28;

          u.x = Math.round(u.x - shiftX);
          u.y = Math.round(u.y - shiftY);
          v.x = Math.round(v.x + shiftX);
          v.y = Math.round(v.y + shiftY);
        }
      }
    }

    // Keep strict clear zone around target at (0, 0)
    const targetClearRadius = baseR1 * 0.76;
    for (let i = 1; i < positionedNodes.length; i++) {
      const u = positionedNodes[i];
      const distToTarget = Math.sqrt(u.x * u.x + u.y * u.y);
      if (distToTarget < targetClearRadius && distToTarget > 0.001) {
        const scale = targetClearRadius / distToTarget;
        u.x = Math.round(u.x * scale);
        u.y = Math.round(u.y * scale);
      }
    }
  }

  // 8. Calculate Bounding Box
  let minX = 0;
  let maxX = 0;
  let minY = 0;
  let maxY = 0;

  positionedNodes.forEach((n) => {
    if (n.x < minX) minX = n.x;
    if (n.x > maxX) maxX = n.x;
    if (n.y < minY) minY = n.y;
    if (n.y > maxY) maxY = n.y;
  });

  return {
    nodes: positionedNodes,
    edges,
    bounds: {
      minX,
      maxX,
      minY,
      maxY,
      width: Math.max(maxX - minX, 100),
      height: Math.max(maxY - minY, 100),
    },
  };
}
