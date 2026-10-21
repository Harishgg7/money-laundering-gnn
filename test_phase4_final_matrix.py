"""
TRACE Phase 4 Final Validation & Release Freeze Matrix
======================================================
Validates all 26 core production criteria:
1. Dashboard metrics (515,088 accounts, 5,078,345 transactions)
2. High-risk account resolution (8054A41C0)
3. Medium-risk account resolution (8000323C0)
4. Low-risk account resolution (800000AC0)
5. Arbitrary account A (803D922C0)
6. Arbitrary account B (8039155C0)
7. Invalid account handling -> HTTP 404
8. Exact account search
9. Partial account search
10. No-result search handling
11. Subject-centered network retrieval
12. Stable deterministic graph layout
13. Risk filtering isolation
14. Account detail retrieval
15. AccountPanel contract compliance
16. Network -> Investigation identity consistency
17. Investigation -> Network identity consistency
18. Dashboard -> Network identity consistency
19. Dynamic report creation for arbitrary account
20. In-place report update without duplicate creation
21. Server-side binary PDF generation
22. PDF account identity consistency
23. Honest SMTP delivery notification
24. Compliance terminology audit (zero prohibited terms)
25. Frozen ML pipeline invariance (checkpoint, graph, threshold)
26. TypeScript and Vite production bundle build
"""

import os
import sys
import json
import urllib.request
import urllib.error
import subprocess

BASE_URL = "http://localhost:8000"

def get(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def get_raw(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, resp.headers.get_content_type(), resp.read()

def get_raw_status(path):
    url = f"{BASE_URL}{path}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"raw": body}
        return e.code, parsed

def post_json(path, data):
    url = f"{BASE_URL}{path}"
    encoded = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=encoded, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def put_json(path, data):
    url = f"{BASE_URL}{path}"
    encoded = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=encoded, headers={"Content-Type": "application/json"}, method="PUT")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def run_matrix():
    print("=" * 80, flush=True)
    print("  TRACE PHASE 4 — FINAL VALIDATION, HARDENING & RELEASE FREEZE MATRIX", flush=True)
    print("=" * 80, flush=True)

    passed = 0
    total = 26

    # 1. Dashboard metrics
    try:
        st, data = get("/api/dashboard")
        assert st == 200
        total_acc = data.get("activeAccounts") or data.get("totalAccounts", 0)
        assert total_acc == 515088
        assert data.get("totalTransactions") == 5078345
        assert len(data.get("topAccounts", [])) > 0
        print(f"[PASS] Test 01: Dashboard live metrics                 | Accounts={total_acc:,}, Txs={data['totalTransactions']:,}", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 01: Dashboard metrics failed: {e}", flush=True)

    # 2. High-risk account
    try:
        st, data = get("/api/network?account=8054A41C0&risk=ALL")
        assert st == 200
        tgt = next((n for n in data["nodes"] if n.get("isTarget")), None)
        assert tgt is not None and tgt["riskLevel"] == "HIGH" and tgt["riskScore"] >= 0.38
        print(f"[PASS] Test 02: High-risk account resolution           | Subject={tgt['accountId']}, Risk={tgt['riskLevel']}, Score={tgt['riskScore']:.4f}", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 02: High-risk resolution failed: {e}", flush=True)

    # 3. Medium-risk account
    try:
        st, data = get("/api/network?account=8000323C0&risk=ALL")
        assert st == 200
        tgt = next((n for n in data["nodes"] if n.get("isTarget")), None)
        assert tgt is not None and tgt["riskLevel"] == "MEDIUM" and 0.20 <= tgt["riskScore"] < 0.38
        print(f"[PASS] Test 03: Medium-risk account resolution         | Subject={tgt['accountId']}, Risk={tgt['riskLevel']}, Score={tgt['riskScore']:.4f}", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 03: Medium-risk resolution failed: {e}", flush=True)

    # 4. Low-risk account
    try:
        st, data = get("/api/network?account=800000AC0&risk=ALL")
        assert st == 200
        tgt = next((n for n in data["nodes"] if n.get("isTarget")), None)
        assert tgt is not None and tgt["riskLevel"] == "LOW" and tgt["riskScore"] < 0.20
        print(f"[PASS] Test 04: Low-risk account resolution            | Subject={tgt['accountId']}, Risk={tgt['riskLevel']}, Score={tgt['riskScore']:.4f}", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 04: Low-risk resolution failed: {e}", flush=True)

    # 5. Arbitrary account A
    try:
        st, data = get("/api/network?account=803D922C0&risk=ALL")
        assert st == 200
        tgt = next((n for n in data["nodes"] if n.get("isTarget")), None)
        assert tgt is not None
        print(f"[PASS] Test 05: Arbitrary account A resolution         | Account=803D922C0, SubgraphNodes={len(data['nodes'])}, Edges={len(data['edges'])}", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 05: Arbitrary account A failed: {e}", flush=True)

    # 6. Arbitrary account B
    try:
        st, data = get("/api/network?account=8039155C0&risk=ALL")
        assert st == 200
        tgt = next((n for n in data["nodes"] if n.get("isTarget")), None)
        assert tgt is not None
        print(f"[PASS] Test 06: Arbitrary account B resolution         | Account=8039155C0, SubgraphNodes={len(data['nodes'])}, Edges={len(data['edges'])}", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 06: Arbitrary account B failed: {e}", flush=True)

    # 7. Invalid account lookup -> 404
    try:
        st, err = get_raw_status("/api/network?account=NONEXISTENT_999999")
        assert st == 404
        assert "not found" in err.get("error", "").lower()
        print(f"[PASS] Test 07: Invalid account handling -> 404        | Code=404, Message='{err['error']}'", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 07: Invalid account handling failed: {e}", flush=True)

    # 8. Exact account search
    try:
        st, data = get("/api/network/search?q=8000323C0&limit=5")
        assert st == 200 and len(data) >= 1 and data[0]["accountId"] == "8000323C0"
        print(f"[PASS] Test 08: Exact account autocomplete search      | Query='8000323C0' -> Match={data[0]['accountId']}, Score={data[0]['riskScore']:.4f}", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 08: Exact search failed: {e}", flush=True)

    # 9. Partial account search
    try:
        st, data = get("/api/network/search?q=8000&limit=5")
        assert st == 200 and len(data) > 0
        print(f"[PASS] Test 09: Partial account prefix search          | Query='8000' -> Returned {len(data)} results, TopMatch={data[0]['accountId']}", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 09: Partial search failed: {e}", flush=True)

    # 10. No-result search
    try:
        st, data = get("/api/network/search?q=NONEXISTENT_XYZ_999&limit=5")
        assert st == 200 and isinstance(data, list) and len(data) == 0
        print(f"[PASS] Test 10: Empty/no-result search handling        | Query='NONEXISTENT_XYZ_999' -> Clean empty array []", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 10: No-result search failed: {e}", flush=True)

    # 11. Subject-centered network retrieval
    try:
        st, data = get("/api/network?account=8000323C0&risk=ALL")
        tgt = next(n for n in data["nodes"] if n.get("isTarget"))
        assert tgt["accountId"] == "8000323C0"
        direct_edges = [e for e in data["edges"] if e["source"] == tgt["accountId"] or e["target"] == tgt["accountId"]]
        assert len(direct_edges) > 0
        print(f"[PASS] Test 11: Subject-centered network retrieval      | Target={tgt['accountId']} (isTarget=True), DirectEdges={len(direct_edges)}", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 11: Subject-centered retrieval failed: {e}", flush=True)

    # 12. Stable deterministic graph layout
    try:
        st1, data1 = get("/api/network?account=8054A41C0&risk=ALL")
        st2, data2 = get("/api/network?account=8054A41C0&risk=ALL")
        assert [n["id"] for n in data1["nodes"]] == [n["id"] for n in data2["nodes"]]
        assert [(e["source"], e["target"]) for e in data1["edges"]] == [(e["source"], e["target"]) for e in data2["edges"]]
        print(f"[PASS] Test 12: Deterministic network layout           | 100% identical node/edge ordering across repeated requests", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 12: Deterministic layout check failed: {e}", flush=True)

    # 13. Risk filtering isolation
    try:
        st_all, data_all = get("/api/network?account=8054A41C0&risk=ALL")
        st_high, data_high = get("/api/network?account=8054A41C0&risk=HIGH")
        assert len(data_high["nodes"]) <= len(data_all["nodes"])
        for n in data_high["nodes"]:
            if not n.get("isTarget"):
                assert n["riskLevel"] == "HIGH"
        print(f"[PASS] Test 13: Risk filter topological isolation      | ALL={len(data_all['nodes'])} nodes -> HIGH={len(data_high['nodes'])} nodes", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 13: Risk filtering failed: {e}", flush=True)

    # 14. Account detail retrieval
    try:
        st, data = get("/api/accounts/8000323C0")
        assert st == 200 and data.get("accountId") == "8000323C0"
        assert "riskScore" in data and "totalVolume" in data
        print(f"[PASS] Test 14: Account detail retrieval endpoint      | Subject={data['accountId']}, NodeId={data['nodeId']}, Volume=${data['totalVolume']:,.2f}", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 14: Account detail retrieval failed: {e}", flush=True)

    # 15. AccountPanel contract compliance
    try:
        st, data = get("/api/accounts/8054A41C0")
        required_fields = ["accountId", "nodeId", "riskScore", "riskLevel", "totalVolume", "transactionCount"]
        for rf in required_fields:
            assert rf in data, f"Missing {rf}"
        print(f"[PASS] Test 15: AccountPanel data contract compliance  | All 6 required fields strictly validated", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 15: AccountPanel contract failed: {e}", flush=True)

    # 16. Network -> Investigation identity
    try:
        st, dossier = get("/api/investigations/8000323C0")
        assert st == 200 and dossier["accountId"] == "8000323C0"
        assert dossier["riskLevel"] == "MEDIUM"
        print(f"[PASS] Test 16: Network -> Investigation identity      | Subject account 8000323C0 maps to dossier {dossier.get('caseId')}", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 16: Network -> Investigation identity failed: {e}", flush=True)

    # 17. Investigation -> Network identity
    try:
        st, dossier = get("/api/investigations/CASE-8000323C0")
        assert st == 200
        subgraph_nodes = dossier.get("subgraph", {}).get("nodes", [])
        tgt = next((n for n in subgraph_nodes if n.get("isTarget") or n.get("accountId") == "8000323C0"), None)
        assert tgt is not None
        print(f"[PASS] Test 17: Investigation -> Network identity      | Dossier subgraph preserves target node 8000323C0", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 17: Investigation -> Network identity failed: {e}", flush=True)

    # 18. Dashboard -> Network identity
    try:
        st, dash = get("/api/dashboard")
        top_acc = dash["topAccounts"][0]["accountId"]
        st_net, net_data = get(f"/api/network?account={top_acc}")
        assert st_net == 200
        tgt = next((n for n in net_data["nodes"] if n.get("isTarget")), None)
        assert tgt is not None and tgt["accountId"] == top_acc
        print(f"[PASS] Test 18: Dashboard -> Network identity          | Top account {top_acc} resolves directly in network explorer", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 18: Dashboard -> Network identity failed: {e}", flush=True)

    # 19. Dynamic report creation
    new_report_id = None
    try:
        payload = {
            "accountId": "8000323C0",
            "caseId": "CASE-8000323C0",
            "riskLevel": "MEDIUM",
            "riskScore": 0.3381,
            "decision": "Continue Surveillance",
            "notes": "Phase 4 release validation: Dynamic report creation test.",
            "status": "Draft",
            "evidence": {
                "transactionCount": 2,
                "totalVolume": 5348.03,
                "connectedAccountsCount": 27
            }
        }
        st, resp = post_json("/api/reports", payload)
        assert st == 201
        new_report_id = resp["id"]
        assert resp["accountId"] == "8000323C0"
        print(f"[PASS] Test 19: Dynamic report creation                 | Assigned Report ID={new_report_id}, Account=8000323C0", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 19: Dynamic report creation failed: {e}", flush=True)

    # 20. In-place report update
    try:
        assert new_report_id is not None
        st_list_before, reports_before = get("/api/reports")
        update_payload = {
            "decision": "Escalate for Further Review",
            "notes": "Phase 4 release validation: Final updated investigator notes."
        }
        st_upd, upd_resp = put_json(f"/api/reports/{new_report_id}", update_payload)
        assert st_upd == 200
        assert upd_resp["decision"] == "Escalate for Further Review"

        st_list_after, reports_after = get("/api/reports")
        assert len(reports_after) == len(reports_before), "Report count changed! Duplicate report created."
        print(f"[PASS] Test 20: In-place report decision & notes update | Updated in-place (no duplication), Decision='{upd_resp['decision']}'", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 20: Report update failed: {e}", flush=True)

    # 21. Server-side binary PDF generation
    try:
        assert new_report_id is not None
        st, ctype, pdf_bytes = get_raw(f"/api/reports/{new_report_id}/pdf")
        assert st == 200
        assert ctype == "application/pdf"
        assert pdf_bytes.startswith(b"%PDF-")
        assert len(pdf_bytes) > 2000
        print(f"[PASS] Test 21: Server-side binary PDF generation       | Status=200, Content-Type=application/pdf, Size={len(pdf_bytes):,} bytes", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 21: PDF generation failed: {e}", flush=True)

    # 22. PDF account identity consistency
    try:
        assert new_report_id is not None
        import io, pdfplumber
        st, ctype, pdf_bytes = get_raw(f"/api/reports/{new_report_id}/pdf")
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pdf_text = "\n".join(p.extract_text() for p in pdf.pages)
        assert new_report_id in pdf_text
        assert "8000323C0" in pdf_text
        assert "Escalate for Further Review" in pdf_text
        print(f"[PASS] Test 22: PDF account identity consistency        | Verified ReportID, AccountID & Decision embedded in PDF binary", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 22: PDF identity consistency failed: {e}", flush=True)

    # 23. Honest SMTP unconfigured notification
    try:
        assert new_report_id is not None
        st, email_resp = post_json(f"/api/reports/{new_report_id}/email", {
            "recipient": "investigator-audit@bank.internal",
            "subject": "TRACE Compliance Package",
            "attachPdf": True
        })
        # If unconfigured, honest status is smtp_unconfigured with success=False
        assert email_resp.get("status") == "smtp_unconfigured"
        assert email_resp.get("success") is False
        print(f"[PASS] Test 23: Honest SMTP boundary handling           | Status='{email_resp.get('status')}', Success={email_resp.get('success')} (honest failure)", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 23: SMTP boundary failed: {e}", flush=True)

    # 24. Compliance terminology audit
    try:
        prohibited = ["conviction score", "conviction", "guilty", "confirmed fraud", "confirmed money laundering"]
        found = []
        for sdir in ["frontend/src", "backend"]:
            for root, _, files in os.walk(sdir):
                for f in files:
                    if f.endswith((".ts", ".tsx", ".py", ".json")) and not f.endswith(("test_phase3_1_matrix.py", "test_phase3_2_matrix.py", "test_phase4_final_matrix.py")):
                        path = os.path.join(root, f)
                        with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                            for lno, line in enumerate(fp, 1):
                                lower = line.lower()
                                for p in prohibited:
                                    if p in lower:
                                        found.append((path, lno, p))
        assert len(found) == 0, f"Found prohibited terms: {found}"
        print(f"[PASS] Test 24: Compliance terminology audit            | Zero prohibited terms across all production code", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 24: Compliance terminology audit failed: {e}", flush=True)

    # 25. Frozen ML pipeline invariance
    try:
        ckpt_path = "checkpoints/best_model.pt"
        graph_path = "data/processed/aml_graph.pt"
        assert os.path.exists(ckpt_path), "Missing model checkpoint"
        assert os.path.exists(graph_path), "Missing aml_graph.pt"
        assert os.path.getsize(ckpt_path) == 511330
        assert os.path.getsize(graph_path) == 68962330
        with open("outputs/enriched_cases.json", "r", encoding="utf-8") as f:
            cases = json.load(f)
            threshold = cases.get("meta", {}).get("threshold") or cases.get("metadata", {}).get("threshold")
            assert threshold == 0.3781
        print(f"[PASS] Test 25: Frozen ML pipeline invariance          | Checkpoint=511,330B, Graph=68,962,330B, Threshold=0.3781", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 25: ML pipeline invariance failed: {e}", flush=True)

    # 26. TypeScript and Vite production build
    try:
        res_tsc = subprocess.run(
            ["node", "node_modules/typescript/bin/tsc", "-b"],
            cwd="frontend",
            capture_output=True,
            text=True
        )
        assert res_tsc.returncode == 0, f"TypeScript errors:\n{res_tsc.stderr or res_tsc.stdout}"

        res_vite = subprocess.run(
            ["node", "node_modules/vite/bin/vite.js", "build"],
            cwd="frontend",
            capture_output=True,
            text=True
        )
        assert res_vite.returncode == 0, f"Vite build errors:\n{res_vite.stderr or res_vite.stdout}"
        print(f"[PASS] Test 26: TypeScript & Vite production build     | tsc -b & vite build passed with 0 errors", flush=True)
        passed += 1
    except Exception as e:
        print(f"[FAIL] Test 26: Build verification failed: {e}", flush=True)

    print("=" * 80, flush=True)
    print(f"  PHASE 4 FINAL MATRIX RESULTS: {passed}/{total} TESTS PASSED ({passed/total*100:.0f}% SUCCESS)", flush=True)
    print("=" * 80, flush=True)

    if passed == total:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    run_matrix()
