"""
Test Runner with Coverage Analysis
=================================
Comprehensive test runner with coverage reporting for the TradingView charts module
"""

import subprocess
import sys
import os
from pathlib import Path
import json


def install_coverage_if_needed():
    """Install coverage package if not available"""
    try:
        import coverage
        print("✅ Coverage package is available")
        return True
    except ImportError:
        print("📦 Installing coverage package...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'coverage'])
            print("✅ Coverage package installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install coverage: {e}")
            return False


def run_tests_with_coverage():
    """Run tests with coverage analysis"""
    
    # Ensure we're in the right directory
    current_dir = Path(__file__).parent
    os.chdir(current_dir)
    
    print("🚀 Starting Test Suite with Coverage Analysis")
    print("=" * 60)
    
    # Install coverage if needed
    if not install_coverage_if_needed():
        print("⚠️ Running tests without coverage analysis")
        return run_tests_only()
    
    # Coverage configuration
    coverage_config = {
        'source': ['tradingview_charts'],
        'omit': [
            '*/tests/*',
            '*/test_*',
            '*/venv/*',
            '*/env/*',
            '*/__pycache__/*',
            '*/backup_files/*',
            '*/results/*',
            '*/models/*'
        ]
    }
    
    try:
        # Start coverage
        print("🔍 Initializing coverage analysis...")
        subprocess.run([
            sys.executable, '-m', 'coverage', 'erase'
        ], check=True)
        
        # Run tests with coverage
        print("🧪 Running unit tests with coverage...")
        result = subprocess.run([
            sys.executable, '-m', 'coverage', 'run',
            '--source=.',
            '--omit=*/tests/*,*/test_*,*/venv/*,*/env/*,*/__pycache__/*',
            '-m', 'pytest', 'tests/unit/test_tradingview_charts.py', '-v'
        ], capture_output=False)
        
        if result.returncode != 0:
            print("⚠️ Some tests failed, but continuing with coverage report...")
        
        # Generate coverage report
        print("\n📊 Generating coverage report...")
        
        # Console report
        subprocess.run([
            sys.executable, '-m', 'coverage', 'report',
            '--show-missing'
        ], check=False)
        
        # HTML report
        try:
            subprocess.run([
                sys.executable, '-m', 'coverage', 'html',
                '-d', 'htmlcov'
            ], check=False)
            print("📄 HTML coverage report generated in 'htmlcov' directory")
        except subprocess.CalledProcessError:
            print("⚠️ Could not generate HTML coverage report")
        
        # JSON report for programmatic access
        try:
            subprocess.run([
                sys.executable, '-m', 'coverage', 'json',
                '-o', 'coverage.json'
            ], check=False)
            
            # Read and display summary
            if os.path.exists('coverage.json'):
                with open('coverage.json', 'r') as f:
                    coverage_data = json.load(f)
                
                total_coverage = coverage_data.get('totals', {}).get('percent_covered', 0)
                print(f"\n🎯 Total Coverage: {total_coverage:.1f}%")
                
                if total_coverage >= 90:
                    print("✅ Coverage target (90%) achieved!")
                else:
                    print(f"⚠️ Coverage below target (90%). Need {90 - total_coverage:.1f}% more.")
                    
        except Exception as e:
            print(f"⚠️ Could not generate JSON coverage report: {e}")
        
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Coverage analysis failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during coverage analysis: {e}")
        return False


def run_tests_only():
    """Run tests without coverage (fallback)"""
    print("🧪 Running tests without coverage analysis...")
    
    try:
        # Try pytest first
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 'tests/unit/test_tradingview_charts.py', '-v'
        ], capture_output=False)
        
        if result.returncode == 0:
            print("✅ All tests passed!")
            return True
        else:
            print("⚠️ Some tests failed")
            return False
            
    except FileNotFoundError:
        # Fallback to unittest
        print("📦 pytest not found, using unittest...")
        try:
            result = subprocess.run([
                sys.executable, 'tests/unit/test_tradingview_charts.py'
            ], capture_output=False)
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            return False


def run_specific_test_class(test_class=None):
    """Run specific test class"""
    if not test_class:
        return run_tests_with_coverage()
    
    print(f"🎯 Running specific test class: {test_class}")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            f'tests/unit/test_tradingview_charts.py::{test_class}', 
            '-v'
        ], capture_output=False)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Failed to run test class {test_class}: {e}")
        return False


def main():
    """Main test runner"""
    print("TradingView Charts - Test Suite Runner")
    print("=" * 50)
    
    # Check if specific test class is requested
    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        success = run_specific_test_class(test_class)
    else:
        success = run_tests_with_coverage()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 TEST SUITE COMPLETED SUCCESSFULLY!")
        print("✅ All tests passed")
    else:
        print("⚠️ TEST SUITE COMPLETED WITH ISSUES")
        print("❌ Some tests failed or coverage below target")
    
    print("=" * 60)
    
    # Additional recommendations
    print("\n📋 RECOMMENDATIONS:")
    print("• Check htmlcov/index.html for detailed coverage report")
    print("• Review any failed tests and fix issues")
    print("• Add more tests for uncovered code paths")
    print("• Ensure all edge cases are tested")
    
    return 0 if success else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)