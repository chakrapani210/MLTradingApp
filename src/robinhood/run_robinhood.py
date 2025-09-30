"""
Robinhood Trading Application Launcher
"""

import sys
import subprocess
import os
from pathlib import Path

def main():
    """Launch the Robinhood trading application"""
    print("🎯 ROBINHOOD TRADING LAUNCHER")
    print("=" * 60)
    
    # Get path to the Robinhood app
    robinhood_root = Path(__file__).parent
    robinhood_app = robinhood_root / "robinhood_app.py"
    
    if not robinhood_app.exists():
        print("❌ Robinhood app not found!")
        print(f"Expected: {robinhood_app}")
        return
    
    # Parse any arguments passed to this script
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    print(f"🚀 Running: python robinhood_app.py {' '.join(args)}")
    print("-" * 60)
    
    # Run the Robinhood app
    try:
        result = subprocess.run(
            [sys.executable, str(robinhood_app)] + args,
            cwd=str(robinhood_root),
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print(f"\n❌ App exited with code: {result.returncode}")
        else:
            print(f"\n✅ App completed successfully!")
            
    except Exception as e:
        print(f"❌ Error running Robinhood app: {e}")

if __name__ == "__main__":
    main()