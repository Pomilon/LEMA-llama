#!/usr/bin/env python3
"""
LEMA Inference Framework Entry Point
"""

import sys
import os
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.engines.cli_engine import CLIChatEngine

def main():
    """Run the chat interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="LEMA Custom Inference Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CLI chat
  python run_chat.py ./checkpoints/final --engine cli
  
  # Run API server (if implemented)
  python run_chat.py ./checkpoints/final --engine api --port 8000
  
  # Run Gradio web UI (if implemented)
  python run_chat.py ./checkpoints/final --engine gradio
        """
    )
    
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to LEMA checkpoint directory"
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["cli", "api", "gradio"],
        default="cli",
        help="Interface engine to use"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda/cpu)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API/Gradio server"
    )
    
    args = parser.parse_args()
    
    if args.engine == "cli":
        engine = CLIChatEngine(args.checkpoint, args.device)
        engine.run()
    
    elif args.engine == "api":
        print("❌ API engine not yet implemented")
        print("Implement in engines/api_engine.py")
        sys.exit(1)
    
    elif args.engine == "gradio":
        print("❌ Gradio engine not yet implemented")
        print("Implement in engines/gradio_engine.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
