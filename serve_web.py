#!/usr/bin/env python3
"""
serve_web.py - Quick server for NBA-VAR web frontend

Serves the web interface on localhost with automatic browser opening.
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path
import argparse


def find_free_port(start_port=8000, max_attempts=10):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socketserver.TCPServer(("", port), None) as s:
                return port
        except OSError:
            continue
    return None


def serve(port=8000, directory="web", open_browser=True):
    """Serve the web app"""
    
    # Change to web directory
    web_dir = Path(directory)
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
    Handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", actual_port), Handler) as httpd:
            url = f"http://localhost:{actual_port}"
            
            print("\n" + "="*60)
            print("NBA-VAR Web Frontend")
            print("="*60)
            print(f"\nServing at: {url}")
            print(f"Directory: {web_dir.absolute()}")
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
    
    args = parser.parse_args()
    
    return serve(
        port=args.port,
        directory=args.dir,
        open_browser=not args.no_browser
    )


if __name__ == "__main__":
    sys.exit(main())
