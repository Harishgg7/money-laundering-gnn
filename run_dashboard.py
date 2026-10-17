"""
AEGIS AML - 1-Click Showcase Dashboard Launcher
Launches the FastAPI backend and opens the interactive dashboard in your browser.
"""
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Check for Python 3.12 with PyG installed
PY312_PATH = Path(r"C:\Users\harish\AppData\Local\Programs\Python\Python312\python.exe")
python_bin = str(PY312_PATH) if PY312_PATH.exists() else sys.executable

print("=" * 76)
print("   AEGIS AML — EXPLAINABLE GRAPH NEURAL NETWORK DASHBOARD")
print("   SRM Institute of Science and Technology · CSE")
print("   Harish GG · Sibins Koshy · Hemakeshwar")
print("=" * 76)
print(f"[*] Python interpreter : {python_bin}")
print(f"[*] Workspace directory: {ROOT}")
print(f"[*] Starting server on : http://127.0.0.1:8000")
print("-" * 76)

server_script = ROOT / "server.py"

# Open browser after a brief delay
def open_browser():
    time.sleep(2)
    print("[*] Opening browser at http://127.0.0.1:8000 ...")
    webbrowser.open("http://127.0.0.1:8000")

import threading
threading.Thread(target=open_browser, daemon=True).start()

try:
    subprocess.run([python_bin, str(server_script)], check=True, cwd=str(ROOT))
except KeyboardInterrupt:
    print("\n[!] Dashboard server stopped.")
