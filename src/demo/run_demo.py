"""
Demo Launcher - Redirects to Unified Paper Trading Demo

This demo folder is kept for organization but now uses the unified demo application.
"""

import sys
import subprocess
import os
from pathlib import Path

def main():
    """Launch the unified paper trading demo"""
    print("🎯 DEMO FOLDER LAUNCHER")
    print("📁 Redirecting to unified paper trading demo...")
    print("=" * 60)
    
    # Get path to the main demo (now in same folder)
    demo_root = Path(__file__).parent
    unified_demo = demo_root / "paper_trading_demo.py"
    
    if not unified_demo.exists():
        print("❌ Unified demo not found!")
        print(f"Expected: {unified_demo}")
        return
    
    # Parse any arguments passed to this script
    args = sys.argv[1:] if len(sys.argv) > 1 else ['production']
    
    print(f"🚀 Running: python paper_trading_demo.py {' '.join(args)}")
    print("-" * 60)
    
    # Run the unified demo
    try:
        result = subprocess.run(
            [sys.executable, str(unified_demo)] + args,
            cwd=str(demo_root),
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print(f"\n❌ Demo exited with code: {result.returncode}")
        else:
            print(f"\n✅ Demo completed successfully!")
            
    except Exception as e:
        print(f"❌ Error running unified demo: {e}")

if __name__ == "__main__":
    main()