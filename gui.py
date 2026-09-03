import os
import sys
import time
import shutil
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
import src.config as config

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8000
APP_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


def start_backend_server():
    """Runs FastAPI server on background loopback thread."""
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")


class DesktopApi:
    """Exposed to the frontend as `window.pywebview.api.*`.

    The embedded webview (pywebview) does not show a native "Save As" dialog
    when the page navigates to a binary file response the way a real browser
    does - clicking a download link just silently fails to render. Since the
    backend and this desktop shell run on the same machine, the report file
    is copied directly from disk instead of round-tripping through HTTP.
    """

    def __init__(self, window):
        self._window = window

    def save_evaluated_report(self, filename: str) -> dict:
        src_path = Path(config.DATA_DIR) / filename
        if not src_path.exists():
            return {"ok": False, "error": f"File not found: {filename}"}

        result = self._window.create_file_dialog(
            webview_save_dialog(),
            save_filename=filename,
            file_types=("Excel Workbook (*.xlsx)",),
        )
        if not result:
            return {"ok": False, "cancelled": True}

        dest_path = result[0] if isinstance(result, (list, tuple)) else result
        try:
            shutil.copyfile(src_path, dest_path)
            return {"ok": True, "path": str(dest_path)}
        except Exception as e:
            return {"ok": False, "error": str(e)}


def webview_save_dialog():
    """Lazily resolves pywebview's SAVE_DIALOG constant (module import is
    deferred - see main())."""
    import webview
    return webview.SAVE_DIALOG


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
        window.expose(DesktopApi(window).save_evaluated_report)
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
