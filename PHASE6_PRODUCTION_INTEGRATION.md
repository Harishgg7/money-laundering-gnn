# TRACE Phase 6 — TRACE-HYBRID-v2 Production Integration & Verification

## 1. Objective

Phase 6 is the final engineering integration and verification gate for **TRACE — AML Investigation & Financial Crime Intelligence**. Following the completion of ML experimentation and multi-dimensional validation in Phase 5, Phase 6 ensures that the officially approved and promoted production model:

> **TRACE-HYBRID-v2**

is actively wired into the live TRACE backend service, REST API endpoints, investigator dossiers, report registry, and server-side PDF generator, without modifying any validated model weights or hyperparameters.

---

## 2. Initial Discrepancy Resolution

During the initial code audit of the repository, a critical discrepancy was identified:
- **Phase 5 Documentation & Manifest:** Declared `TRACE-HYBRID-v2` as the promoted production model with operating threshold `0.9184`.
- **Live Implementation (`trace_service.py`):** Still defaulted to loading `checkpoints/best_model.pt` (the topology-only GraphSAGE baseline with operating threshold `0.3781`).
- **Resolution:** `trace_service.py` was refactored to directly instantiate `AdvancedGraphSAGE` with `use_residual=False`, load the 128-dimensional encoder (`models/gnn_encoder.pt`), load the 150-tree XGBoost classifier (`models/xgb_classifier.json`), mount the 29 engineered AML features (`experiments/ml_improvement/extended_features.pt`), and enforce the official threshold `0.9184`.

---

## 3. Changes Made

| File | Change Description | Rationale |
|---|---|---|
| `model.py` | Added `ResSAGEConvBlock` and `AdvancedGraphSAGE` module definitions. | Eliminates external script dependencies; enables native instantiation of the 3-layer GraphSAGE encoder. |
| `trace_service.py` | 1. Defined `AML_29_FEATURE_NAMES`.<br>2. Added production hybrid initialization (`TRACE-HYBRID-v2`, threshold `0.9184`).<br>3. Implemented on-demand `predict_hybrid(node_idx)` using local 3-hop neighborhood extraction.<br>4. Reconciled risk level classification and added `set_model()` for runtime switching/rollback. | Replaces baseline GraphSAGE inference with the validated hybrid model across all account queries, network graph extraction, and dossiers. |
| `server.py` | 1. Added `/api/model_info` endpoint returning live model metadata and manifest.<br>2. Updated report creation to embed `modelVersion` and `threshold`. | Exposes live model provenance to investigators and ensures generated reports record official metadata. |
| `report_pdf.py` | Updated default operating threshold to `0.9184` and updated Section 4 to reflect `TRACE-HYBRID-v2`. | Ensures generated PDF compliance packages document the correct production model and operating threshold. |
| `model_manifest.json` | Root copy placed alongside `models/model_manifest.json`. | Guarantees consistent manifest discovery from both service and test runners. |
| `test_phase6_production_integration.py` | Created comprehensive 17-test integration verification suite. | Programmatically verifies artifacts, hashes, architectures, inference, APIs, workflows, and terminology. |

---

## 4. Production Inference Architecture

The live production inference pipeline executes on-demand without relying solely on cached lookup tables:

```
                      Target Account Node
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
       2-Hop / 3-Hop Subgraph         29 AML Behavioral &
      Topology (PyG edge_index)       Transaction Features
               │                               │
               ▼                               │
       3-Layer GraphSAGE                       │
       Relational Encoder                      │
               │                               │
               ▼                               │
     128-dim Graph Embedding                   │
               │                               │
               └───────────────┬───────────────┘
                               ▼
               157-dim Combined Representation
                               │
                               ▼
                    XGBoost Classifier
                   (150 Boosted Trees)
                               │
                               ▼
                     Model Risk Probability
                            [0, 1]
                               │
                               ▼
                   Threshold Comparison (0.9184)
                               │
                               ▼
                Investigator Decision-Support:
                • Model Risk Score
                • Risk Level (HIGH / MEDIUM / LOW)
                • Feature Contributions
```

---

## 5. Model Verification & Artifact Provenance

### Production Model: TRACE-HYBRID-v2

- **Model Version:** `TRACE-HYBRID-v2`
- **Architecture:** 3-layer GraphSAGE (128-dim embeddings) + 29 AML behavioral features + XGBoost classifier
- **Input Dimensions:** 128 (graph embedding) + 29 (engineered tabular features) = 157 total inputs
- **Operating Threshold:** `0.9184` (F1-maximizing validation threshold)
- **Official Metrics (Phase 5 Held-Out Test Set):**
  - **PR-AUC:** `0.4569` (+33.6% lift over baseline)
  - **ROC-AUC:** `0.9412` (+3.5% lift over baseline)
  - **Precision:** `71.89%` (+26.1% absolute improvement over baseline)
  - **Recall:** `37.00%`
  - **F1 Score:** `0.4886`

### Cryptographic Artifact SHA256 Checksums

| Artifact | Path | Size (Bytes) | SHA256 Checksum |
|---|---|---:|---|
| **GNN Encoder** | `models/gnn_encoder.pt` | 375,586 | `3d0049901334e1d3dd4a529f1110938b53eb897500a24c18ebd3e3818e6c9d35` |
| **XGBoost Classifier** | `models/xgb_classifier.json` | 794,061 | `4b8e555608122e2c6cc406d087b2a2e975a2da733e2e72997b198b9c77099f21` |
| **Verified Probabilities** | `models/hybrid_probs.pt` | 2,061,557 | `74d1aafbfbd455519e9bcbc3c23202cd8585c076f04640be9aa82c8abfd5a7d4` |
| **Rollback Checkpoint** | `checkpoints/best_model.pt` | 511,330 | `8d37c392f865d2a76404c8e416222f742e406e87580a98dcd96f2c6692aed543` |

---

## 6. Integration Test Suite (Phase 6)

The dedicated Phase 6 suite (`test_phase6_production_integration.py`) validates all 17 critical criteria:

| Test ID | Verification Criterion | Status | Verified Result |
|---|---|:---:|---|
| **Test 01** | Production Model Artifacts Exist | **PASS** | GNN, XGBoost, Probabilities, and Manifest present |
| **Test 02** | Manifest Consistency | **PASS** | `TRACE-HYBRID-v2`, `PROMOTED_PRODUCTION`, threshold `0.9184` |
| **Test 03** | Cryptographic Artifact Hashes | **PASS** | 100% SHA256 match with `model_manifest.json` |
| **Test 04** | GNN Encoder Loads | **PASS** | 3-layer `AdvancedGraphSAGE` (29 -> 128 -> 128) |
| **Test 05** | XGBoost Loads | **PASS** | 150 boosted estimators, 157 input features |
| **Test 06** | 29 Engineered AML Features | **PASS** | Exactly 29 features across 515,088 accounts |
| **Test 07** | 128-Dimensional Graph Embedding | **PASS** | GraphSAGE output vector dimension = 128 |
| **Test 08** | Hybrid Inference Output | **PASS** | Sample probability in `[0.0, 1.0]` (`0.2171`) |
| **Test 09** | Deterministic Inference | **PASS** | Bit-exact output across consecutive runs (`0.821120`) |
| **Test 10** | Operating Threshold | **PASS** | Exactly `0.9184` |
| **Test 11** | Model Version Provenance | **PASS** | Identifies as `"TRACE-HYBRID-v2"` |
| **Test 12** | Safe Boundary Rejection | **PASS** | Unknown accounts reject safely via `KeyError` / 404 |
| **Test 13** | Rollback Model Preserved | **PASS** | `TRACE-GSAGE-v1` at `checkpoints/best_model.pt` intact |
| **Test 14** | Live API Endpoint Integration | **PASS** | `/api/model_info` & `/api/accounts/8054A41C0` return hybrid data |
| **Test 15** | Investigation Dossier Integration | **PASS** | `/api/investigations/{id}` records `TRACE-HYBRID-v2` & `0.9184` |
| **Test 16** | Report and PDF Integration | **PASS** | Generated report & binary PDF reflect `TRACE-HYBRID-v2` |
| **Test 17** | Strict Compliance Terminology Audit | **PASS** | Zero prohibited terms across backend, frontend & reports |

**Result:** `17/17 TESTS PASSED (100% SUCCESS)`.

---

## 7. Regression Validation Results

All previously established test suites were executed against the live application:

- **Phase 4 Final Validation Matrix:** `26/26 TESTS PASSED (100% SUCCESS)`
  - Dashboard live metrics (515,088 accounts, 5,078,345 transactions)
  - Account resolution (`8054A41C0`, `8000323C0`, `800000AC0`, `803D922C0`, `8039155C0`)
  - Full-graph autocomplete search & no-result handling
  - Subject-centered bounded network extraction
  - AccountPanel contracts & identity consistency across views
  - In-place report update & server-side PDF generation
  - Honest unconfigured SMTP boundary handling
  - Frozen ML pipeline invariance (checkpoint, graph size)
  - TypeScript compilation & Vite production bundle build
- **Phase 5 ML Integrity Suite:** `10/10 TESTS PASSED (100% SUCCESS)`
  - Artifact existence, checksums, feature dimensions, embedding dimensions, XGBoost tree counts, deterministic inference, and rollback preservation.

---

## 8. Investigator Workflow End-to-End Verification

The complete investigator journey was verified across the live application:

1. **Dashboard:** Live KPIs display 515,088 active accounts, 5,078,345 transactions, volume trends, and suspicious accounts list.
2. **Network Explorer:** Bounded 2-hop / 3-hop neighborhood renders with smooth interactive layout; selected subject accounts display consistent hybrid scores.
3. **Account Panel:** Slide-over detail panel reflects `TRACE-HYBRID-v2`, threshold `0.9184`, and calibrated risk scores.
4. **Investigation Dossier:** Complete dossier loads with multi-hop topological graph, 29 behavioral features, and XGBoost feature contributions.
5. **Report Creation & Adjudication:** Investigator decisions ("File Investigation Report", "Escalate for Further Review", etc.) persist cleanly in `outputs/reports_registry.json`.
6. **PDF Generation:** Server-side ReportLab engine renders binary PDF including `TRACE-HYBRID-v2`, threshold `0.9184`, observed transaction patterns, quantitative evidence, and compliance adjudication.

---

## 9. Rollback Architecture

The baseline model is fully preserved for regulatory compliance and operational fallback:
- **Baseline Model:** `TRACE-GSAGE-v1`
- **Location:** `checkpoints/best_model.pt`
- **Size:** `511,330` bytes (exact byte count verified)
- **SHA256:** `8d37c392f865d2a76404c8e416222f742e406e87580a98dcd96f2c6692aed543`
- **Operating Threshold:** `0.3781`
- **Runtime Switching:** `TraceService.set_model("TRACE-GSAGE-v1")` provides an in-process model switch (where supported by the server instance), while setting the `TRACE_MODEL_VERSION=TRACE-GSAGE-v1` environment variable configures the active model on startup (which requires restarting the server process). Both approaches preserve the rollback model without modifying code.

---

## 10. Known Project Limitations

1. **Graph Extraction Bounding:** Neighborhood extraction remains bounded to 2-hop / 3-hop local subgraphs and a maximum node limit to ensure low latency and browser stability.
2. **Read-Only Sourced Data:** The underlying IBM AML dataset is externally sourced and read-only.
3. **SMTP Notification:** External notification relies on SMTP server configuration; unconfigured environments return an honest `smtp_unconfigured` status.
4. **Decision-Support Scope:** The system provides risk assessment and decision-support intelligence, not automated legal or criminal determinations.

---

## 11. Final Status

```text
PRODUCTION INTEGRATION VERIFIED
```
