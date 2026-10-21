"""
run_dashboard.py — One-command launcher for TRACE AML Investigation Platform.
Builds enriched cases if missing and launches the local web server.
"""

import os
import sys
import argparse
import subprocess
import webbrowser
from pathlib import Path
from threading import Timer

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

ROOT_DIR = Path(__file__).resolve().parent

from build_enriched_cases import build_cases
from server import start_server



def main():
    parser = argparse.ArgumentParser(description="Launch TRACE AML Investigation Dashboard")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on (default: 8000)")
    parser.add_argument("--rebuild-cases", action="store_true", help="Force regeneration of outputs/enriched_cases.json")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser tab")
    args = parser.parse_args()

    cases_path = ROOT_DIR / "outputs" / "enriched_cases.json"
    if args.rebuild_cases or not os.path.exists(cases_path):
        try:
            build_cases()
        except Exception as e:
            print(f"[TRACE] Continuing with pre-embedded verified HI-Small cases ({e})")

    if not args.no_browser:
        Timer(1.0, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()

    start_server(port=args.port)


if __name__ == "__main__":
    main()