#!/usr/bin/env python3
"""
Serve the built multi-sport static site locally with clean-route support.
"""

from __future__ import annotations

import argparse
import http.server
import socket
import sys
import webbrowser
from functools import partial
from http import HTTPStatus
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


DEFAULT_BIND_HOST = "127.0.0.1"
SCRIPT_PATH = Path(__file__).resolve()
SITE_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_WEB_DIR = SITE_ROOT / "dist"


class RobustThreadingHTTPServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def resolve_bind_host(host: str) -> str | None:
    try:
        return socket.gethostbyname(host.strip())
    except socket.gaierror:
        return None


def find_free_port(host: str, start_port: int = 8000, max_attempts: int = 10) -> int | None:
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((host, port))
                return port
        except OSError:
            continue
    return None


class MultiPageRequestHandler(http.server.SimpleHTTPRequestHandler):
    server_version = "SportsWorkspaceHTTP"
    sys_version = ""

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Permissions-Policy",
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), microphone=(), payment=(), usb=()",
        )
        super().end_headers()

    def list_directory(self, path):
        self.send_error(HTTPStatus.NOT_FOUND, "Directory listing is disabled")
        return None

    def _is_safe_relative_path(self, relative_path: str) -> bool:
        base_dir = Path(self.directory).resolve()
        candidate = (base_dir / relative_path).resolve()
        try:
            candidate.relative_to(base_dir)
        except ValueError:
            return False
        return True

    def _normalize_clean_route(self, request_path: str) -> str | None:
        parsed = urlsplit(request_path or "/")
        path = unquote(parsed.path or "/")

        if not path.startswith("/"):
            path = f"/{path}"
        if "\\" in path or "\x00" in path:
            return None

        normalized_parts = PurePosixPath(path).parts
        if any(part == ".." for part in normalized_parts):
            return None

        if path == "/":
            return "/index.html"

        if "." in PurePosixPath(path).name:
            return path

        slug = path.rstrip("/")
        candidate = f"{slug.lstrip('/')}.html"
        if self._is_safe_relative_path(candidate) and (Path(self.directory) / candidate).is_file():
            return f"/{candidate}"

        return path

    def _prepare_request(self) -> str | None:
        normalized = self._normalize_clean_route(self.path)
        if normalized is None:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid request path")
            return None
        return normalized

    def do_GET(self):
        normalized = self._prepare_request()
        if normalized is None:
            return
        self.path = normalized
        return super().do_GET()

    def do_HEAD(self):
        normalized = self._prepare_request()
        if normalized is None:
            return
        self.path = normalized
        return super().do_HEAD()

    def _method_not_allowed(self) -> None:
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Allow", "GET, HEAD")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        self._method_not_allowed()

    def do_PUT(self):
        self._method_not_allowed()

    def do_DELETE(self):
        self._method_not_allowed()

    def do_PATCH(self):
        self._method_not_allowed()

    def do_OPTIONS(self):
        self._method_not_allowed()


def serve(
    port: int = 8000,
    directory: str = str(DEFAULT_WEB_DIR),
    host: str = DEFAULT_BIND_HOST,
    open_browser: bool = True,
) -> int:
    if not (1 <= int(port) <= 65535):
        print(f"Error: port must be between 1 and 65535, got {port}")
        return 1

    web_dir = Path(directory).resolve()
    if not web_dir.is_dir():
        print(f"Error: directory '{directory}' does not exist")
        print("Build the site first with: python sports/site/pipeline/build_static_site.py")
        return 1

    bind_host = resolve_bind_host(host)
    if bind_host is None:
        print(f"Error: could not resolve host '{host}'")
        return 1

    actual_port = find_free_port(bind_host, start_port=port)
    if actual_port is None:
        print(f"Error: could not find a free port on {bind_host} starting from {port}")
        return 1

    if actual_port != port:
        print(f"Port {port} is busy, using port {actual_port} instead")

    handler_factory = partial(MultiPageRequestHandler, directory=str(web_dir))
    sport_dirs = sorted(path.name for path in web_dir.iterdir() if path.is_dir() and (path / "index.html").exists())

    try:
        with RobustThreadingHTTPServer((bind_host, actual_port), handler_factory) as httpd:
            browser_host = "localhost" if bind_host in {"0.0.0.0", "127.0.0.1"} else host
            url = f"http://{browser_host}:{actual_port}"

            print("\n" + "=" * 60)
            print("Multi-Sport Analytics Site")
            print("=" * 60)
            print(f"\nServing at: {url}")
            print(f"Bound to: {bind_host}:{actual_port}")
            print(f"Directory: {web_dir}")
            if sport_dirs:
                print(f"Sports: {', '.join('/' + name for name in sport_dirs)}")
            print("\nPress Ctrl+C to stop the server")
            print("=" * 60 + "\n")

            if open_browser:
                try:
                    webbrowser.open(url)
                except webbrowser.Error as exc:
                    print(f"Warning: could not open browser automatically: {exc}")

            httpd.serve_forever()

    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        return 0
    except Exception as exc:
        print(f"\nError: {exc}")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve the built multi-sport analytics site",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sports/site/pipeline/serve_web.py
  python sports/site/pipeline/serve_web.py --port 3000
  python sports/site/pipeline/serve_web.py --host 0.0.0.0
  python sports/site/pipeline/serve_web.py --no-browser
        """,
    )
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to serve on (default: 8000)")
    parser.add_argument(
        "--dir",
        "-d",
        type=str,
        default=str(DEFAULT_WEB_DIR),
        help=f"Directory to serve (default: {DEFAULT_WEB_DIR})",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_BIND_HOST,
        help=f"Host/interface to bind (default: {DEFAULT_BIND_HOST})",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    args = parser.parse_args()

    return serve(
        port=args.port,
        directory=args.dir,
        host=args.host,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    sys.exit(main())
