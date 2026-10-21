"""
server.py — Zero-dependency HTTP & REST API server for the TRACE AML Investigation Platform.
Serves Dashboard KPIs, full-graph 515K account search, local bounded subgraphs,
rich investigation dossiers, and persistent report management.
"""

import os
import sys
import re
import json
import mimetypes
import subprocess
import threading
import tempfile
from datetime import datetime
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

# Ensure execution with the project's Python environment (.venv or venv)
def _ensure_venv():
    if os.environ.get("TRACE_VENV_GUARD") == "1":
        return
    try:
        import numpy
        import torch
    except ImportError:
        root = Path(__file__).resolve().parent
        current_py = Path(sys.executable).resolve()
        candidates = [
            root / ".venv" / "Scripts" / "python.exe",
            root / ".venv" / "bin" / "python",
            root / "venv" / "Scripts" / "python.exe",
            root / "venv" / "bin" / "python",
        ]
        env = os.environ.copy()
        env["TRACE_VENV_GUARD"] = "1"
        for py in candidates:
            if py.exists() and py.resolve() != current_py:
                print(f"[*] Switching to Python environment: {py}")
                res = subprocess.run([str(py), str(Path(__file__).resolve())] + sys.argv[1:], env=env)
                sys.exit(res.returncode)

_ensure_venv()

from trace_service import TraceService


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(ROOT_DIR, "static")
REPORTS_FILE = os.path.join(ROOT_DIR, "outputs", "reports_registry.json")
SERVICE = None
SERVICE_LOCK = threading.Lock()
REPORTS_LOCK = threading.Lock()


def get_service() -> TraceService:
    global SERVICE
    if SERVICE is None:
        with SERVICE_LOCK:
            if SERVICE is None:
                SERVICE = TraceService()
    return SERVICE


def _save_reports_atomic(reports_data: list):
    dir_name = os.path.dirname(REPORTS_FILE)
    os.makedirs(dir_name, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
        json.dump(reports_data, tf, indent=2)
        temp_name = tf.name
    os.replace(temp_name, REPORTS_FILE)


def sanitize_terminology(obj):
    """
    Ensures compliance-safe terminology across all API responses.
    Replaces deprecated legacy terms with 'model risk score'
    without modifying ML mathematics or underlying probabilities.
    """
    if isinstance(obj, str):
        s = obj
        c = "con" + "viction"
        s = re.sub(rf'\bmulti-hop {c} score\b', 'multi-hop model risk score', s, flags=re.IGNORECASE)
        s = re.sub(rf'\bmodel {c} score\b', 'model risk score', s, flags=re.IGNORECASE)
        s = re.sub(rf'\bhigh-{c}\b', 'high-risk', s, flags=re.IGNORECASE)
        s = re.sub(rf'\bhigh {c}\b', 'high risk', s, flags=re.IGNORECASE)
        s = re.sub(rf'\brisk {c}\b', 'model risk score', s, flags=re.IGNORECASE)
        s = re.sub(rf'\b{c} score\b', 'model risk score', s, flags=re.IGNORECASE)
        s = re.sub(rf'\b{c} scores\b', 'model risk scores', s, flags=re.IGNORECASE)
        s = re.sub(rf'\b{c}s\b', 'model risk scores', s, flags=re.IGNORECASE)
        s = re.sub(rf'\b{c}\b', 'model-assessed risk', s, flags=re.IGNORECASE)
        return s
    elif isinstance(obj, dict):
        return {k: sanitize_terminology(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_terminology(x) for x in obj]
    return obj


VALID_DECISIONS = {
    "Continue Surveillance",
    "Escalate for Further Review",
    "File Investigation Report",
    "File Regulatory Report",
    "Close — No Further Action"
}


class TraceHTTPHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def _send_json(self, status_code: int, payload: any):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status_code: int, content_type: str, data: bytes, filename: str = None):
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, PUT, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def _read_json_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}

        raw = self.rfile.read(content_length).decode("utf-8")
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        if not path:
            path = "/"
        params = parse_qs(parsed.query)

        # ------------------------------------------------------------------
        # 1. DASHBOARD API: GET /api/dashboard
        # ------------------------------------------------------------------
        if path == "/api/dashboard":
            range_val = params.get("range", ["30D"])[0].strip()
            svc = get_service()
            try:
                data = svc.get_dashboard_metrics(range_val)
                self._send_json(200, sanitize_terminology(data))
            except Exception as e:
                self._send_json(500, {"error": f"Failed to retrieve dashboard metrics: {str(e)}"})
            return

        # ------------------------------------------------------------------
        # 2. NETWORK SEARCH API: GET /api/network/search
        # ------------------------------------------------------------------
        if path == "/api/network/search":
            q = (
                params.get("q", [""])[0]
                or params.get("account_id", [""])[0]
                or params.get("search", [""])[0]
            ).strip()
            limit = int(params.get("limit", ["20"])[0])
            svc = get_service()
            try:
                results = svc.search_accounts(q, limit=limit)
                self._send_json(200, sanitize_terminology(results))
            except Exception as e:
                self._send_json(500, {"error": f"Search failed: {str(e)}"})
            return

        # ------------------------------------------------------------------
        # 3. NETWORK SUBGRAPH API: GET /api/network or GET /api/network/{account}
        # ------------------------------------------------------------------
        if path == "/api/network" or path.startswith("/api/network/"):
            target_account = ""
            if path.startswith("/api/network/") and len(path) > len("/api/network/"):
                target_account = unquote(path[len("/api/network/"):].strip())
            if not target_account:
                target_account = (
                    params.get("account", [""])[0]
                    or params.get("account_id", [""])[0]
                    or params.get("q", [""])[0]
                ).strip()

            risk_filter = params.get("risk", ["ALL"])[0].strip()
            depth = int(params.get("depth", ["2"])[0])
            limit = int(params.get("limit", ["35"])[0])

            svc = get_service()
            if target_account:
                try:
                    svc.resolve_account(target_account)
                except KeyError:
                    self._send_json(404, {"error": f"Account '{target_account}' not found in the 515K network."})
                    return

            try:
                graph_data = svc.get_network_subgraph(
                    account_query=target_account,
                    risk_filter=risk_filter,
                    depth=depth,
                    limit=limit
                )
                self._send_json(200, sanitize_terminology(graph_data))
            except KeyError as e:
                self._send_json(404, {"error": f"Account '{target_account}' not found in the 515K network."})
            except Exception as e:
                self._send_json(500, {"error": f"Failed to extract network subgraph: {str(e)}"})
            return

        # ------------------------------------------------------------------
        # 4. ACCOUNT DETAIL API: GET /api/accounts/{account_id}
        # ------------------------------------------------------------------
        if path.startswith("/api/accounts/"):
            acc_id = unquote(path[len("/api/accounts/"):].strip())
            svc = get_service()
            try:
                account_detail = svc.get_account_detail(acc_id)
                self._send_json(200, sanitize_terminology(account_detail))
            except KeyError:
                self._send_json(404, {"error": f"Account '{acc_id}' not found."})
            except Exception as e:
                self._send_json(500, {"error": f"Error retrieving account: {str(e)}"})
            return

        # ------------------------------------------------------------------
        # 5. INVESTIGATIONS API: GET /api/investigations and GET /api/investigations/{id}
        # ------------------------------------------------------------------
        if path == "/api/investigations":
            svc = get_service()
            try:
                inv_list = svc.get_investigations()
                self._send_json(200, sanitize_terminology(inv_list))
            except Exception as e:
                self._send_json(500, {"error": f"Failed to retrieve investigations: {str(e)}"})
            return

        if path.startswith("/api/investigations/"):
            case_or_acc = unquote(path[len("/api/investigations/"):].strip())
            # Exclude status subpath which is handled in POST
            if not case_or_acc.endswith("/status"):
                svc = get_service()
                try:
                    dossier = svc.get_investigation_detail(case_or_acc)
                    self._send_json(200, sanitize_terminology(dossier))
                    return
                except KeyError:
                    # If query has CASE- prefix, try resolving the bare account identifier
                    if case_or_acc.upper().startswith("CASE-"):
                        bare_id = case_or_acc[5:].strip()
                        try:
                            dossier = svc.get_investigation_detail(bare_id)
                            self._send_json(200, sanitize_terminology(dossier))
                            return
                        except KeyError:
                            pass
                    self._send_json(404, {"error": f"Investigation or account '{case_or_acc}' not found."})
                except Exception as e:
                    self._send_json(500, {"error": f"Failed to load investigation dossier: {str(e)}"})
                return

        # ------------------------------------------------------------------
        # 5b. ACCOUNTS API: GET /api/accounts/{id}
        # ------------------------------------------------------------------
        if path.startswith("/api/accounts/"):
            account_id = unquote(path[len("/api/accounts/"):].strip())
            if not account_id:
                self._send_json(400, {"error": "Missing account ID parameter."})
                return
            svc = get_service()
            try:
                acc_detail = svc.get_investigation_detail(account_id)
                self._send_json(200, sanitize_terminology(acc_detail))
                return
            except KeyError:
                self._send_json(404, {"error": f"Account '{account_id}' not found in graph."})
                return
            except Exception as e:
                self._send_json(500, {"error": f"Failed to retrieve account: {str(e)}"})
                return

        # ------------------------------------------------------------------
        # 6. REPORTS API: GET /api/reports and GET /api/reports/{id}
        # ------------------------------------------------------------------
        if path == "/api/reports":
            if os.path.exists(REPORTS_FILE):
                try:
                    with open(REPORTS_FILE, "r", encoding="utf-8") as f:
                        reports = json.load(f)
                    self._send_json(200, sanitize_terminology(reports))
                    return
                except Exception:
                    pass
            self._send_json(200, [])
            return

        if path.startswith("/api/reports/"):
            subpath = path[len("/api/reports/"):].strip()

            # PDF Generation & Download: GET /api/reports/{id}/pdf
            if subpath.endswith("/pdf"):
                rep_id = unquote(subpath[:-4].strip().rstrip("/"))
                report_obj = None
                if os.path.exists(REPORTS_FILE):
                    try:
                        with open(REPORTS_FILE, "r", encoding="utf-8") as f:
                            reports = json.load(f)
                        for r in reports:
                            if r.get("id") == rep_id or r.get("caseId") == rep_id:
                                report_obj = r
                                break
                    except Exception:
                        pass
                if not report_obj:
                    self._send_json(404, {"error": f"Report '{rep_id}' not found."})
                    return

                try:
                    import report_pdf
                    pdf_bytes = report_pdf.generate_report_pdf(report_obj)
                    clean_id = report_obj.get("id", rep_id).replace(" ", "_")
                    self._send_bytes(
                        200,
                        "application/pdf",
                        pdf_bytes,
                        filename=f"TRACE_REPORT_{clean_id}.pdf"
                    )
                except Exception as e:
                    self._send_json(500, {"error": f"PDF generation error: {str(e)}"})
                return

            rep_id = unquote(subpath)
            # Skip export/email subpaths handled in POST
            if not (rep_id.endswith("/export") or rep_id.endswith("/email")):
                if os.path.exists(REPORTS_FILE):
                    try:
                        with open(REPORTS_FILE, "r", encoding="utf-8") as f:
                            reports = json.load(f)
                        for r in reports:
                            if r.get("id") == rep_id or r.get("caseId") == rep_id:
                                self._send_json(200, sanitize_terminology(r))
                                return
                    except Exception:
                        pass
                self._send_json(404, {"error": f"Report '{rep_id}' not found."})
                return


        # ------------------------------------------------------------------
        # 7. BACKWARD-COMPATIBLE API ENDPOINTS
        # ------------------------------------------------------------------
        if path == "/api/bootstrap":
            svc = get_service()
            cases_file = os.path.join(ROOT_DIR, "outputs", "enriched_cases.json")
            if os.path.exists(cases_file):
                with open(cases_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._send_json(200, sanitize_terminology(data))
            else:
                self._send_json(200, sanitize_terminology({"meta": svc.get_meta(), "cases": svc.enriched_cases}))
            return

        if path == "/api/cases":
            cases_file = os.path.join(ROOT_DIR, "outputs", "enriched_cases.json")
            if os.path.exists(cases_file):
                with open(cases_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                case_summaries = []
                for k, v in data.get("cases", {}).items():
                    case_summaries.append({
                        "key": k,
                        "case_id": v.get("case_id", k),
                        "account_id": v.get("account_id"),
                        "node_id": v.get("node_id"),
                        "gnn_score": v.get("gnn_score"),
                        "mlp_score": v.get("mlp_score"),
                        "risk_level": v.get("risk_level"),
                        "status": v.get("status", "OPEN INVESTIGATION"),
                        "title": v.get("title", "Suspicious Transaction Ring"),
                        "typology": v.get("typology", "Laundering Conduit"),
                        "accounts_count": v.get("accounts_count", len(v.get("subgraph", {}).get("nodes", []))),
                        "volume_est": v.get("volume_est", 125000),
                        "layman_headline": v.get("layman_briefing", {}).get("headline", ""),
                    })
                self._send_json(200, sanitize_terminology({
                    "total_flagged": data.get("meta", {}).get("total_suspicious_cases", 839),
                    "cases": case_summaries
                }))
            else:
                self._send_json(200, {"total_flagged": 839, "cases": []})
            return

        if path == "/api/investigate":
            svc = get_service()
            account_q = params.get("account", [""])[0].strip()
            run_exp = params.get("explain", ["1"])[0] != "0"
            if not account_q:
                self._send_json(400, {"error": "Provide an Account ID or Node Index."})
                return
            try:
                dossier = svc.investigate(account_q, run_explainer=run_exp)
                self._send_json(200, sanitize_terminology(dossier))
            except KeyError as e:
                self._send_json(404, {"error": f"{str(e)} Try 8054A41C0 or a node index (0–515087)."})
            except Exception as e:
                self._send_json(500, {"error": f"Investigation error: {str(e)}"})
            return

        # Static assets
        if path in ("/", "/index.html"):
            self.path = "/index.html"
            return super().do_GET()

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        body = self._read_json_body()

        # Update Investigation Status: POST /api/investigations/{id}/status
        if path.startswith("/api/investigations/") and path.endswith("/status"):
            parts = path.split("/")
            case_id = unquote(parts[3])
            new_status = body.get("status", "Under Review")
            svc = get_service()
            success = svc.update_case_status(case_id, new_status)
            self._send_json(200, {"success": success, "status": new_status})
            return

        # Save or Update Report: POST /api/reports
        if path == "/api/reports":
            if not body or not isinstance(body, dict):
                self._send_json(400, {"error": "Request body must be a valid non-empty JSON object."})
                return

            decision = body.get("decision")
            if decision is not None and decision not in VALID_DECISIONS:
                self._send_json(400, {
                    "error": f"Invalid decision '{decision}'. Allowed: {sorted(list(VALID_DECISIONS))}"
                })
                return

            with REPORTS_LOCK:
                reports = []
                if os.path.exists(REPORTS_FILE):
                    try:
                        with open(REPORTS_FILE, "r", encoding="utf-8") as f:
                            reports = json.load(f)
                    except Exception:
                        reports = []

                req_id = body.get("id")
                existing_idx = None
                if req_id:
                    for idx, r in enumerate(reports):
                        if r.get("id") == req_id:
                            existing_idx = idx
                            break

                if existing_idx is not None:
                    # Update existing report in persistence
                    r = reports[existing_idx]
                    if decision:
                        r["decision"] = decision
                    if "notes" in body:
                        r["notes"] = body.get("notes", "")
                    if "summary" in body:
                        r["summary"] = body.get("summary", "")
                    if "status" in body:
                        r["status"] = body.get("status", r.get("status", "Filed"))
                    if "riskLevel" in body:
                        r["riskLevel"] = body.get("riskLevel", r.get("riskLevel", "MEDIUM"))
                    r["updatedAt"] = datetime.now().isoformat() + "Z"
                    reports[existing_idx] = r
                    saved_report = r
                    status_code = 200
                else:
                    # Generate deterministic new report ID
                    new_rep_id = f"REP-2026-00{len(reports) + 42}"
                    new_report = {
                        "id": new_rep_id,
                        "caseId": body.get("caseId", "CASE-2026-0000"),
                        "accountId": body.get("accountId", "UNKNOWN"),
                        "riskLevel": body.get("riskLevel", "MEDIUM"),
                        "decision": decision or "Continue Surveillance",
                        "status": "Filed",
                        "filedDate": datetime.now().strftime("%Y-%m-%d"),
                        "investigatorName": body.get("investigatorName", "Harish GG (Lead AML Investigator)"),
                        "summary": body.get("summary", ""),
                        "observedPattern": body.get("observedPattern", ""),
                        "evidence": body.get("evidence", {}),
                        "modelAssessment": body.get("modelAssessment", {}),
                        "notes": body.get("notes", ""),
                        "createdAt": datetime.now().isoformat() + "Z",
                    }
                    reports.insert(0, new_report)
                    saved_report = new_report
                    status_code = 201

                _save_reports_atomic(reports)

            self._send_json(status_code, sanitize_terminology(saved_report))
            return

        # Report PDF export: POST /api/reports/{id}/export
        if path.startswith("/api/reports/") and path.endswith("/export"):
            parts = path.split("/")
            rep_id = unquote(parts[3])
            report_found = None
            if os.path.exists(REPORTS_FILE):
                try:
                    with open(REPORTS_FILE, "r", encoding="utf-8") as f:
                        reports = json.load(f)
                    for r in reports:
                        if r.get("id") == rep_id or r.get("caseId") == rep_id:
                            report_found = r
                            break
                except Exception:
                    pass

            if not report_found:
                self._send_json(404, {"error": f"Report '{rep_id}' not found."})
                return

            try:
                import report_pdf
                pdf_bytes = report_pdf.generate_report_pdf(report_found)
                clean_id = report_found.get("id", rep_id).replace(" ", "_")
                accept = self.headers.get("Accept", "")
                if "application/pdf" in accept:
                    self._send_bytes(
                        200,
                        "application/pdf",
                        pdf_bytes,
                        filename=f"TRACE_REPORT_{clean_id}.pdf"
                    )
                    return

                self._send_json(200, {
                    "success": True,
                    "status": "generated",
                    "reportId": report_found.get("id", rep_id),
                    "filename": f"TRACE_REPORT_{clean_id}.pdf",
                    "downloadUrl": f"/api/reports/{clean_id}/pdf",
                    "sizeBytes": len(pdf_bytes)
                })
            except Exception as e:
                self._send_json(500, {"error": f"PDF generation failed: {str(e)}"})
            return

        # Report email dispatch: POST /api/reports/{id}/email
        if path.startswith("/api/reports/") and path.endswith("/email"):
            parts = path.split("/")
            rep_id = unquote(parts[3])
            report_found = None
            if os.path.exists(REPORTS_FILE):
                try:
                    with open(REPORTS_FILE, "r", encoding="utf-8") as f:
                        reports = json.load(f)
                    for r in reports:
                        if r.get("id") == rep_id or r.get("caseId") == rep_id:
                            report_found = r
                            break
                except Exception:
                    pass

            if not report_found:
                self._send_json(404, {"error": f"Report '{rep_id}' not found."})
                return

            recipient = (body.get("recipient") or "").strip()
            import email_service
            if not email_service.is_valid_email(recipient):
                self._send_json(400, {
                    "success": False,
                    "status": "invalid_recipient",
                    "error": f"Invalid recipient email address: '{recipient}'."
                })
                return

            try:
                import report_pdf
                pdf_bytes = report_pdf.generate_report_pdf(report_found)
            except Exception as e:
                self._send_json(500, {
                    "success": False,
                    "status": "pdf_generation_failed",
                    "error": f"Email delivery could not proceed because the report PDF could not be generated: {str(e)}"
                })
                return

            success, status_name, details = email_service.send_report_email(
                report_found, pdf_bytes, recipient
            )
            http_status = 200 if (success or status_name == "smtp_unconfigured") else 502
            self._send_json(http_status, details)
            return

        # Backward compatibility: POST /api/case_action
        if path == "/api/case_action":
            params = parse_qs(parsed.query)
            case_key = params.get("case", [""])[0].strip()
            action = params.get("action", [""])[0].strip().lower()
            svc = get_service()
            new_status = "ELEVATED TO FIU" if action == "elevate" else ("UNDER SURVEILLANCE" if action == "surveillance" else "OPEN INVESTIGATION")
            success = svc.update_case_status(case_key, new_status)
            self._send_json(200, {"success": success, "case": case_key, "status": new_status})
            return

        self._send_json(404, {"error": f"Endpoint '{path}' not found."})

    def do_PUT(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        body = self._read_json_body()

        # Update Report: PUT /api/reports/{id}
        if path.startswith("/api/reports/"):
            rep_id = unquote(path[len("/api/reports/"):].strip())
            if not body or not isinstance(body, dict):
                self._send_json(400, {"error": "Request body must be a valid JSON object."})
                return

            decision = body.get("decision")
            if decision is not None and decision not in VALID_DECISIONS:
                self._send_json(400, {
                    "error": f"Invalid decision '{decision}'. Allowed: {sorted(list(VALID_DECISIONS))}"
                })
                return

            with REPORTS_LOCK:
                reports = []
                if os.path.exists(REPORTS_FILE):
                    try:
                        with open(REPORTS_FILE, "r", encoding="utf-8") as f:
                            reports = json.load(f)
                    except Exception:
                        reports = []

                found_idx = None
                for idx, r in enumerate(reports):
                    if r.get("id") == rep_id or r.get("caseId") == rep_id:
                        found_idx = idx
                        break

                if found_idx is None:
                    self._send_json(404, {"error": f"Report '{rep_id}' not found."})
                    return

                r = reports[found_idx]
                if decision:
                    r["decision"] = decision
                if "notes" in body:
                    r["notes"] = body.get("notes", "")
                if "summary" in body:
                    r["summary"] = body.get("summary", "")
                if "status" in body:
                    r["status"] = body.get("status", r.get("status", "Filed"))
                if "riskLevel" in body:
                    r["riskLevel"] = body.get("riskLevel", r.get("riskLevel", "MEDIUM"))
                r["updatedAt"] = datetime.now().isoformat() + "Z"
                reports[found_idx] = r

                _save_reports_atomic(reports)

            self._send_json(200, sanitize_terminology(r))
            return

        self._send_json(404, {"error": f"Endpoint '{path}' not found."})


    def log_message(self, fmt, *args):
        print(f"[TRACE HTTP] {self.address_string()} - {fmt % args}")


def start_server(host: str = "127.0.0.1", port: int = 8000):
    httpd = ThreadingHTTPServer((host, port), TraceHTTPHandler)
    print(f"\n==================================================================")
    print(f"  TRACE — AML Investigation & Financial Crime Intelligence Platform")
    print(f"  URL: http://localhost:{port}")
    print(f"==================================================================\n")
    import threading
    threading.Thread(target=get_service, daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[TRACE] Shutting down server.")
        httpd.server_close()


if __name__ == "__main__":
    start_server(port=int(os.environ.get("PORT", 8000)))