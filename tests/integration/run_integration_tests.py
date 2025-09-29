#!/usr/bin/env python3
"""
Enhanced Trading System Integration Test Runner

This script provides easy ways to run integration tests and demos
for the Enhanced Trading System Orchestrator.
"""

import sys
import os
import subprocess
import argparse

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.insert(0, project_root)

def run_integration_tests(verbose=False):
    """Run full integration test suite"""
    print("="*80)
    print("RUNNING ENHANCED ORCHESTRATOR INTEGRATION TESTS")
    print("="*80)
    
    test_file = os.path.join(current_dir, "test_enhanced_orchestrator_integration.py")
    
    cmd = [sys.executable, test_file]
    if verbose:
        cmd.append("--verbose")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running integration tests: {e}")
        return False

def run_demo():
    """Run standalone demo"""
    print("="*80)
    print("RUNNING ENHANCED FEATURES DEMO")
    print("="*80)
    
    test_file = os.path.join(current_dir, "test_enhanced_orchestrator_integration.py")
    
    cmd = [sys.executable, test_file, "--demo"]
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running demo: {e}")
        return False

def run_specific_test(test_name, verbose=False):
    """Run a specific test method"""
    print("="*80)
    print(f"RUNNING SPECIFIC TEST: {test_name}")
    print("="*80)
    
    test_file = os.path.join(current_dir, "test_enhanced_orchestrator_integration.py")
    
    cmd = [sys.executable, "-m", "unittest", f"test_enhanced_orchestrator_integration.TestEnhancedOrchestratorIntegration.{test_name}"]
    if verbose:
        cmd.append("-v")
    
    try:
        result = subprocess.run(cmd, cwd=current_dir, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running specific test: {e}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced Trading System Integration Test Runner')
    parser.add_argument('--demo', action='store_true', 
                       help='Run enhanced features demo')
    parser.add_argument('--test', type=str,
                       help='Run specific test (e.g., test_01_system_initialization)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--all', action='store_true',
                       help='Run all integration tests (default)')
    
    args = parser.parse_args()
    
    success = True
    
    if args.demo:
        success = run_demo()
    elif args.test:
        success = run_specific_test(args.test, args.verbose)
    else:
        # Default: run all integration tests
        success = run_integration_tests(args.verbose)
    
    if success:
        print("\n" + "="*80)
        print("INTEGRATION TESTING COMPLETED SUCCESSFULLY!")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("INTEGRATION TESTING FAILED!")
        print("="*80)
        sys.exit(1)

if __name__ == "__main__":
    main()