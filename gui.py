import os
import sys
import time
import threading
import webbrowser
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent
site_packages = project_root / "myenv" / "lib" / "python3.14" / "site-packages"
if site_packages.exists() and str(site_packages) not in sys.path:
    sys.path.insert(0, str(site_packages))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import uvicorn
from src.web_server import app

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
APP_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


def start_backend_server():
    """Runs FastAPI server on background loopback thread."""
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")


def main():
    print("=" * 60)
    print(" VIVI AUTO-EVAL DESKTOP APPLICATION TOOL")
    print("=" * 60)

    # 1. Start FastAPI server thread
    server_thread = threading.Thread(target=start_backend_server, daemon=True)
    server_thread.start()
    print(f"Server initialized at {APP_URL}")
    time.sleep(1.2)

    # 2. Try launching native desktop application window via pywebview
    try:
        import webview
        print("Launching native desktop application shell...")
        window = webview.create_window(
            title="VIVI Auto-Eval Desktop Tool",
            url=APP_URL,
            width=1380,
            height=900,
            resizable=True,
            min_size=(1024, 700)
        )
        webview.start(debug=False)
    except Exception as e:
        print(f"Desktop GUI window notice ({e}). Opening application interface in browser at {APP_URL}...")
        webbrowser.open(APP_URL)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down desktop application server...")


if __name__ == "__main__":
    main()
