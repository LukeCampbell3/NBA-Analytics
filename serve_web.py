#!/usr/bin/env python3
"""
serve_web.py - Quick server for NBA-VAR web frontend

Serves the web interface on localhost with automatic browser opening.
"""

import os
import argparse
import http.server
import json
import socketserver
import sys
import webbrowser
from pathlib import Path
from urllib.parse import urlparse


def find_free_port(start_port=8000, max_attempts=10):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socketserver.TCPServer(("", port), None) as s:
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

    def _normalize_clean_route(self, request_path: str) -> str:
        parsed = urlparse(request_path or "/")
        path = parsed.path or "/"

        if path == "/":
            return "/index.html"

        # Let normal static file requests pass through.
        if "." in Path(path).name:
            return path

        # Support clean routes for html pages.
        slug = path.rstrip("/")
        candidate = f"{slug}.html"
        if candidate.startswith("/"):
            candidate = candidate[1:]
        if Path(candidate).exists():
            return f"/{candidate}"

        # No clean-route match; keep default behavior (404/static handling).
        return path

    def _serve_preloaded_payload(self, normalized_path: str) -> bool:
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
        self.wfile.write(payload)
        return True

    def do_GET(self):
        normalized = self._normalize_clean_route(self.path)
        if self._serve_preloaded_payload(normalized):
            return
        self.path = normalized
        return super().do_GET()


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
    college_cards_path = data_dir / "college_cards.json"
    college_vals_path = data_dir / "college_valuations.json"

    if nba_cards_path.exists():
        payloads["data/cards.json"] = nba_cards_path.read_bytes()
    if nba_vals_path.exists():
        payloads["data/valuations.json"] = nba_vals_path.read_bytes()

    if college_cards_path.exists():
        # Fast path: preload raw bytes when no trimming is requested.
        if college_card_limit is None or college_card_limit <= 0:
            payloads["data/college_cards.json"] = college_cards_path.read_bytes()
            if college_vals_path.exists():
                payloads["data/college_valuations.json"] = college_vals_path.read_bytes()
        else:
            cards = json.loads(college_cards_path.read_text(encoding="utf-8"))
            vals = []
            if college_vals_path.exists():
                vals = json.loads(college_vals_path.read_text(encoding="utf-8"))

            original_count = len(cards) if isinstance(cards, list) else 0
            if isinstance(cards, list):
                if original_count > college_card_limit:
                    cards = sorted(cards, key=card_value_score, reverse=True)[:college_card_limit]
                    keep_keys = {player_identity_key(card) for card in cards}
                    if isinstance(vals, list):
                        vals = [v for v in vals if player_identity_key(v) in keep_keys]
                    print(
                        f"[preload] College payload trimmed: {original_count} -> {len(cards)} cards "
                        f"(limit={college_card_limit})"
                    )
                else:
                    print(f"[preload] College payload under limit: {original_count} cards")

            payloads["data/college_cards.json"] = json.dumps(
                cards if isinstance(cards, list) else [],
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
            payloads["data/college_valuations.json"] = json.dumps(
                vals if isinstance(vals, list) else [],
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")

    return payloads


def serve(port=8000, directory="web", open_browser=True, college_card_limit=1200):
    """Serve the web app"""
    
    # Change to web directory
    web_dir = Path(directory).resolve()
    if not web_dir.exists():
        print(f"Error: Directory '{directory}' does not exist")
        return 1
    
    os.chdir(web_dir)
    
    # Check if data exists
    data_dir = Path("data")
    if not data_dir.exists() or not (data_dir / "cards.json").exists():
        print("\n" + "="*60)
        print("WARNING: No data files found!")
        print("="*60)
        print("\nThe web app will load but show example data only.")
        print("\nTo use real data:")
        print("  1. Generate cards: python create_cards.py --input data.csv --output data/cards")
        print("  2. Run analysis: python analyze_players.py --cards data/cards --output analysis")
        print("  3. Run valuation: python value_players.py --cards data/cards --output valuations")
        print("  4. Prepare web data: python prepare_web_data.py")
        print("\n" + "="*60 + "\n")
    
    # Find free port
    actual_port = find_free_port(port)
    if actual_port is None:
        print(f"Error: Could not find a free port starting from {port}")
        return 1
    
    if actual_port != port:
        print(f"Port {port} is busy, using port {actual_port} instead")
    
    # Create server
    Handler = MultiPageRequestHandler
    effective_limit = None if (college_card_limit is None or college_card_limit <= 0) else int(college_card_limit)
    Handler.json_payload_cache = preload_web_payloads(web_dir=web_dir, college_card_limit=effective_limit)
    if Handler.json_payload_cache:
        preloaded_keys = ", ".join(sorted(Handler.json_payload_cache.keys()))
        print(f"[preload] Loaded JSON payloads at startup: {preloaded_keys}")

    # Discover available pages to make adding future pages obvious.
    pages = sorted(
        p.stem for p in web_dir.glob("*.html")
        if p.is_file()
    )
    
    try:
        with socketserver.TCPServer(("", actual_port), Handler) as httpd:
            url = f"http://localhost:{actual_port}"
            
            print("\n" + "="*60)
            print("NBA-VAR Web Frontend")
            print("="*60)
            print(f"\nServing at: {url}")
            print(f"Directory: {web_dir.absolute()}")
            if pages:
                print(f"Pages: {', '.join('/' + p for p in pages)}")
            print("\nPress Ctrl+C to stop the server")
            print("="*60 + "\n")
            
            # Open browser
            if open_browser:
                print("Opening browser...")
                webbrowser.open(url)
            
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
        description="Serve NBA-VAR web frontend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python serve_web.py                    # Serve on port 8000
  python serve_web.py --port 3000        # Serve on port 3000
  python serve_web.py --no-browser       # Don't open browser
  python serve_web.py --dir web          # Serve from 'web' directory
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
        default='web',
        help='Directory to serve (default: web)'
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
        open_browser=not args.no_browser,
        college_card_limit=args.college_card_limit,
    )


if __name__ == "__main__":
    sys.exit(main())
