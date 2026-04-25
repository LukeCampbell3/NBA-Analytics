#!/usr/bin/env python3
"""
serve_web.py - Quick server for the NBA web frontend

Serves the web interface on localhost with automatic browser opening.
"""

import argparse
import http.server
import json
import socket
import sys
import webbrowser
from functools import partial
from http import HTTPStatus
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


DEFAULT_BIND_HOST = "127.0.0.1"
SCRIPT_PATH = Path(__file__).resolve()
NBA_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_WEB_DIR = NBA_ROOT / "web"


class RobustThreadingHTTPServer(http.server.ThreadingHTTPServer):
    """Threaded HTTP server with sensible local-dev defaults."""

    daemon_threads = True
    allow_reuse_address = True


def resolve_bind_host(host: str) -> str | None:
    """Resolve hostnames to an IPv4 address for binding."""
    try:
        return socket.gethostbyname(host.strip())
    except socket.gaierror:
        return None


def find_free_port(host, start_port=8000, max_attempts=10):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((host, port))
                return port
        except OSError:
            continue
    return None


class MultiPageRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Static file handler with clean-page routes.

    Examples:
      /          -> /index.html
      /about     -> /about.html (if file exists)
      /about/    -> /about.html (if file exists)
    """

    # Populated at server startup (path -> bytes), e.g. "data/cards.json".
    json_payload_cache = {}
    server_version = "NBAAnalyticsHTTP"
    sys_version = ""

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header(
            "Permissions-Policy",
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
            "microphone=(), payment=(), usb=()",
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

        # Let normal static file requests pass through.
        if "." in PurePosixPath(path).name:
            return path

        # Support clean routes for html pages.
        slug = path.rstrip("/")
        candidate = f"{slug.lstrip('/')}.html"
        if self._is_safe_relative_path(candidate) and (Path(self.directory) / candidate).is_file():
            return f"/{candidate}"

        # No clean-route match; keep default behavior (404/static handling).
        return path

    def _serve_preloaded_payload(self, normalized_path: str, send_body: bool = True) -> bool:
        cache_key = normalized_path.lstrip("/")
        payload = self.json_payload_cache.get(cache_key)
        if payload is None:
            return False

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        # Short cache to reduce redundant fetch/parsing when navigating pages.
        self.send_header("Cache-Control", "public, max-age=120")
        self.end_headers()
        if send_body:
            self.wfile.write(payload)
        return True

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
        if self._serve_preloaded_payload(normalized):
            return
        self.path = normalized
        return super().do_GET()

    def do_HEAD(self):
        normalized = self._prepare_request()
        if normalized is None:
            return
        if self._serve_preloaded_payload(normalized, send_body=False):
            return
        self.path = normalized
        return super().do_HEAD()

    def _method_not_allowed(self):
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


def player_identity_key(record):
    """Stable key for mapping cards <-> valuations."""
    player = record.get("player", {}) if isinstance(record, dict) else {}
    pid = str(player.get("id", "")).strip()
    if pid:
        return f"id:{pid}"
    name = str(player.get("name", "")).strip().lower()
    season = str(player.get("season", "")).strip()
    team = str(player.get("team", "")).strip().lower()
    return f"name:{name}|season:{season}|team:{team}"


def card_value_score(card):
    """Sortable player value score with safe fallback."""
    if not isinstance(card, dict):
        return 0.0
    metrics = card.get("value_metrics", {})
    try:
        return float(metrics.get("player_value_score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def read_binary_payload(path: Path, label: str) -> bytes | None:
    """Read a payload file without crashing startup."""
    try:
        return path.read_bytes()
    except OSError as exc:
        print(f"[preload] Warning: could not read {label} from {path}: {exc}")
        return None


def load_json_list(path: Path, label: str) -> list:
    """Load a JSON array with safe fallback."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except OSError as exc:
        print(f"[preload] Warning: could not read {label} from {path}: {exc}")
        return []
    except json.JSONDecodeError as exc:
        print(f"[preload] Warning: invalid JSON in {label} at {path}: {exc}")
        return []

    if isinstance(data, list):
        return data

    print(f"[preload] Warning: expected {label} to contain a JSON array, got {type(data).__name__}")
    return []


def preload_web_payloads(web_dir: Path, college_card_limit: int | None):
    """
    Preload JSON payloads into memory for faster first fetch and cross-page nav.
    Returns dict keyed by relative path (e.g. "data/cards.json") -> bytes payload.
    """
    payloads = {}
    data_dir = web_dir / "data"
    if not data_dir.exists():
        return payloads

    nba_cards_path = data_dir / "cards.json"
    nba_vals_path = data_dir / "valuations.json"
    daily_predictions_path = data_dir / "daily_predictions.json"
    college_cards_path = data_dir / "college_cards.json"
    college_vals_path = data_dir / "college_valuations.json"

    if nba_cards_path.exists():
        nba_cards = read_binary_payload(nba_cards_path, "NBA cards")
        if nba_cards is not None:
            payloads["data/cards.json"] = nba_cards
    if nba_vals_path.exists():
        nba_vals = read_binary_payload(nba_vals_path, "NBA valuations")
        if nba_vals is not None:
            payloads["data/valuations.json"] = nba_vals
    if daily_predictions_path.exists():
        daily_predictions = read_binary_payload(daily_predictions_path, "daily predictions")
        if daily_predictions is not None:
            payloads["data/daily_predictions.json"] = daily_predictions

    if college_cards_path.exists():
        # Fast path: preload raw bytes when no trimming is requested.
        if college_card_limit is None or college_card_limit <= 0:
            college_cards = read_binary_payload(college_cards_path, "college cards")
            if college_cards is not None:
                payloads["data/college_cards.json"] = college_cards
            if college_vals_path.exists():
                college_vals = read_binary_payload(college_vals_path, "college valuations")
                if college_vals is not None:
                    payloads["data/college_valuations.json"] = college_vals
        else:
            cards = load_json_list(college_cards_path, "college cards")
            vals = load_json_list(college_vals_path, "college valuations") if college_vals_path.exists() else []

            original_count = len(cards)
            if original_count > college_card_limit:
                cards = sorted(cards, key=card_value_score, reverse=True)[:college_card_limit]
                keep_keys = {player_identity_key(card) for card in cards}
                vals = [v for v in vals if player_identity_key(v) in keep_keys]
                print(
                    f"[preload] College payload trimmed: {original_count} -> {len(cards)} cards "
                    f"(limit={college_card_limit})"
                )
            else:
                print(f"[preload] College payload under limit: {original_count} cards")

            payloads["data/college_cards.json"] = json.dumps(
                cards,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            payloads["data/college_valuations.json"] = json.dumps(
                vals,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")

    return payloads


def serve(
    port=8000,
    directory=str(DEFAULT_WEB_DIR),
    host=DEFAULT_BIND_HOST,
    open_browser=True,
    college_card_limit=1200,
):
    """Serve the web app"""

    if not (1 <= int(port) <= 65535):
        print(f"Error: Port must be between 1 and 65535, got {port}")
        return 1

    web_dir = Path(directory).resolve()
    if not web_dir.is_dir():
        print(f"Error: Directory '{directory}' does not exist")
        return 1

    bind_host = resolve_bind_host(host)
    if bind_host is None:
        print(f"Error: Could not resolve host '{host}'")
        return 1

    # Check if data exists
    data_dir = web_dir / "data"
    if not data_dir.exists() or not (data_dir / "cards.json").exists():
        print("\n" + "="*60)
        print("WARNING: No data files found!")
        print("="*60)
        print("\nThe web app will load but show example data only.")
        print("\nTo use real data:")
        print("  1. Generate cards: python create_cards.py --input data.csv --output data/cards")
        print("  2. Run analysis: python analyze_players.py --cards data/cards --output analysis")
        print("  3. Run valuation: python value_players.py --cards data/cards --output valuations")
        print("  4. Prepare web data: python sports/nba/pipeline/prepare_web_data.py")
        print("\n" + "="*60 + "\n")

    # Find free port
    actual_port = find_free_port(bind_host, start_port=port)
    if actual_port is None:
        print(f"Error: Could not find a free port on {bind_host} starting from {port}")
        return 1

    if actual_port != port:
        print(f"Port {port} is busy, using port {actual_port} instead")

    # Create server
    effective_limit = None if (college_card_limit is None or college_card_limit <= 0) else int(college_card_limit)
    MultiPageRequestHandler.json_payload_cache = preload_web_payloads(
        web_dir=web_dir,
        college_card_limit=effective_limit,
    )
    if MultiPageRequestHandler.json_payload_cache:
        preloaded_keys = ", ".join(sorted(MultiPageRequestHandler.json_payload_cache.keys()))
        print(f"[preload] Loaded JSON payloads at startup: {preloaded_keys}")
    handler_factory = partial(MultiPageRequestHandler, directory=str(web_dir))

    # Discover available pages to make adding future pages obvious.
    pages = sorted(
        p.stem for p in web_dir.glob("*.html")
        if p.is_file()
    )

    try:
        with RobustThreadingHTTPServer((bind_host, actual_port), handler_factory) as httpd:
            browser_host = "localhost" if bind_host in {"0.0.0.0", "127.0.0.1"} else host
            url = f"http://{browser_host}:{actual_port}"

            print("\n" + "="*60)
            print("NBA Analytics Web Frontend")
            print("="*60)
            print(f"\nServing at: {url}")
            print(f"Bound to: {bind_host}:{actual_port}")
            print(f"Directory: {web_dir}")
            if pages:
                print(f"Pages: {', '.join('/' + p for p in pages)}")
            print("\nPress Ctrl+C to stop the server")
            print("="*60 + "\n")

            # Open browser
            if open_browser:
                print("Opening browser...")
                try:
                    webbrowser.open(url)
                except webbrowser.Error as exc:
                    print(f"Warning: Could not open browser automatically: {exc}")

            # Serve forever
            httpd.serve_forever()

    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        return 0
    except Exception as e:
        print(f"\nError: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Serve NBA web frontend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python serve_web.py                    # Serve on 127.0.0.1:8000
  python serve_web.py --port 3000        # Serve on port 3000
  python serve_web.py --host 0.0.0.0     # Expose on your local network
  python serve_web.py --no-browser       # Don't open browser
  python serve_web.py --dir ../web       # Serve from a custom directory
        """
    )
    
    parser.add_argument(
        '--port', '-p',
        type=int,
        default=8000,
        help='Port to serve on (default: 8000)'
    )
    
    parser.add_argument(
        '--dir', '-d',
        type=str,
        default=str(DEFAULT_WEB_DIR),
        help=f"Directory to serve (default: {DEFAULT_WEB_DIR})"
    )

    parser.add_argument(
        '--host',
        type=str,
        default=DEFAULT_BIND_HOST,
        help=f"Host/interface to bind (default: {DEFAULT_BIND_HOST})"
    )

    parser.add_argument(
        '--no-browser',
        action='store_true',
        help='Do not open browser automatically'
    )
    parser.add_argument(
        '--college-card-limit',
        type=int,
        default=1200,
        help='Max number of college cards to serve (default: 1200, use 0 to disable limit).'
    )
    
    args = parser.parse_args()
    
    return serve(
        port=args.port,
        directory=args.dir,
        host=args.host,
        open_browser=not args.no_browser,
        college_card_limit=args.college_card_limit,
    )


if __name__ == "__main__":
    sys.exit(main())
