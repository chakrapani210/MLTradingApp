#!/usr/bin/env python3
"""
Code Coverage Analysis for ML Trading System
Analyzes the codebase and generates coverage report
"""
import os
import sys
from pathlib import Path
import subprocess
import json

def analyze_codebase_structure():
    """Analyze the codebase structure and identify files to test"""
    project_root = Path.cwd()
    src_dir = project_root / "src"
    
    python_files = []
    
    # Find all Python files in src directory
    for file_path in src_dir.rglob("*.py"):
        if "__pycache__" not in str(file_path) and file_path.name != "__init__.py":
            python_files.append(file_path)
    
    print("📁 Python Files in src/ directory:")
    print("=" * 60)
    
    file_info = {}
    total_lines = 0
    
    for file_path in sorted(python_files):
        relative_path = file_path.relative_to(project_root)
        
        # Count lines
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                code_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
                line_count = len(code_lines)
                total_lines += line_count
                
                # Identify classes and functions
                classes = []
                functions = []
                
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('class ') and ':' in stripped:
                        class_name = stripped.split('class ')[1].split('(')[0].split(':')[0].strip()
                        classes.append(class_name)
                    elif stripped.startswith('def ') and ':' in stripped:
                        func_name = stripped.split('def ')[1].split('(')[0].strip()
                        if not func_name.startswith('_'):  # Focus on public methods
                            functions.append(func_name)
                
                file_info[str(relative_path)] = {
                    'lines': line_count,
                    'classes': classes,
                    'functions': functions
                }
                
                print(f"📄 {relative_path}")
                print(f"   Lines: {line_count}")
                if classes:
                    print(f"   Classes: {', '.join(classes[:3])}{'...' if len(classes) > 3 else ''}")
                if functions:
                    print(f"   Functions: {', '.join(functions[:3])}{'...' if len(functions) > 3 else ''}")
                print()
                
        except Exception as e:
            print(f"   ❌ Error reading file: {e}")
    
    print(f"📊 Total Lines of Code: {total_lines}")
    return file_info, total_lines


def identify_testing_gaps():
    """Identify files that need more testing coverage"""
    print("\n🔍 Testing Gap Analysis:")
    print("=" * 60)
    
    test_dir = Path.cwd() / "tests" / "unit"
    existing_test_files = set()
    
    # Find existing test files
    if test_dir.exists():
        for test_file in test_dir.glob("test_*.py"):
            existing_test_files.add(test_file.name)
    
    print(f"📋 Existing Test Files: {len(existing_test_files)}")
    for test_file in sorted(existing_test_files):
        print(f"   ✅ {test_file}")
    
    # Identify gaps
    src_modules = [
        "enhanced_orchestrator.py",
        "tradingview_charts.py",
        "main.py",
        "data/providers.py",
        "data/preprocessors.py", 
        "models/enhanced_model_management.py",
        "signals/technical.py",
        "backtesting/enhanced_backtesting.py",
        "analysis/enhanced_market_analysis.py",
        "trading/trading_app_base.py",
        "utils/plotting_tools.py"
    ]
    
    needed_tests = []
    for module in src_modules:
        module_name = module.replace("/", "_").replace(".py", "")
        expected_test = f"test_{module_name}.py"
        
        if expected_test not in existing_test_files:
            needed_tests.append((module, expected_test))
    
    print(f"\n📝 Tests Needed: {len(needed_tests)}")
    for module, test_file in needed_tests:
        print(f"   🔲 {test_file} (for {module})")
    
    return needed_tests


def estimate_coverage():
    """Estimate current test coverage"""
    print("\n📈 Coverage Estimation:")
    print("=" * 60)
    
    # Count test assertions
    test_files = list(Path.cwd().glob("tests/unit/test_*.py"))
    test_files.extend([Path.cwd() / "run_simple_tests.py"])
    
    total_tests = 0
    total_assertions = 0
    
    for test_file in test_files:
        if test_file.exists():
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Count test methods
                    test_methods = content.count('def test_')
                    total_tests += test_methods
                    
                    # Count assertions
                    assertions = (content.count('assert') + 
                                content.count('assertEqual') + 
                                content.count('assertTrue') + 
                                content.count('assertFalse') + 
                                content.count('assertIn') + 
                                content.count('assertNotNone'))
                    total_assertions += assertions
                    
                    print(f"📄 {test_file.name}: {test_methods} tests, {assertions} assertions")
                    
            except Exception as e:
                print(f"❌ Error reading {test_file}: {e}")
    
    print(f"\n📊 Test Statistics:")
    print(f"   Total Test Methods: {total_tests}")
    print(f"   Total Assertions: {total_assertions}")
    print(f"   Avg Assertions per Test: {total_assertions/max(total_tests, 1):.1f}")
    
    # Estimate coverage based on files tested vs total files
    file_info, total_lines = analyze_codebase_structure()
    tested_files = len([f for f in test_files if f.exists()])
    total_src_files = len(file_info)
    
    estimated_coverage = (tested_files / max(total_src_files, 1)) * 100
    
    print(f"\n🎯 Estimated Coverage:")
    print(f"   Files Tested: {tested_files}/{total_src_files}")
    print(f"   Estimated Coverage: {estimated_coverage:.1f}%")
    
    return estimated_coverage


def generate_testing_recommendations():
    """Generate recommendations for reaching 90% coverage"""
    print("\n💡 Recommendations for 90% Coverage:")
    print("=" * 60)
    
    recommendations = [
        {
            "priority": "HIGH",
            "action": "Create comprehensive unit tests for signal generators",
            "files": ["tests/unit/test_signal_generators.py"],
            "impact": "25% coverage boost"
        },
        {
            "priority": "HIGH", 
            "action": "Add thorough testing for data providers and model management",
            "files": ["tests/unit/test_data_and_models.py"],
            "impact": "20% coverage boost"
        },
        {
            "priority": "MEDIUM",
            "action": "Test risk management and backtesting components",
            "files": ["tests/unit/test_risk_and_backtest.py"],
            "impact": "15% coverage boost"
        },
        {
            "priority": "MEDIUM",
            "action": "Add integration tests for trading workflows",
            "files": ["tests/integration/test_trading_workflows.py"],
            "impact": "10% coverage boost"
        },
        {
            "priority": "LOW",
            "action": "Test utility functions and plotting tools",
            "files": ["tests/unit/test_utilities.py"],
            "impact": "5% coverage boost"
        }
    ]
    
    for rec in recommendations:
        print(f"🔥 {rec['priority']} PRIORITY: {rec['action']}")
        print(f"   Files: {', '.join(rec['files'])}")
        print(f"   Expected Impact: {rec['impact']}")
        print()
    
    print("📋 Next Steps:")
    print("1. ✅ Basic orchestrator tests (COMPLETED)")
    print("2. 🔄 Signal generator comprehensive tests (IN PROGRESS)")
    print("3. 🔲 Data provider and model management tests")
    print("4. 🔲 Risk management and backtesting tests")
    print("5. 🔲 Integration test workflows")
    
    return recommendations


def run_pytest_coverage():
    """Try to run pytest with coverage if available"""
    print("\n🧪 Attempting to run pytest coverage analysis:")
    print("=" * 60)
    
    try:
        # Try to run pytest with coverage on simple tests first
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "run_simple_tests.py",
            "--verbose"
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Simple tests passed!")
            print(result.stdout)
        else:
            print("⚠️ Simple tests had issues:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Test execution timed out")
    except Exception as e:
        print(f"❌ Error running tests: {e}")


def main():
    """Main analysis function"""
    print("🚀 ML Trading System - Code Coverage Analysis")
    print("=" * 60)
    
    # Analyze codebase structure
    file_info, total_lines = analyze_codebase_structure()
    
    # Identify testing gaps
    needed_tests = identify_testing_gaps()
    
    # Estimate current coverage
    current_coverage = estimate_coverage()
    
    # Generate recommendations
    recommendations = generate_testing_recommendations()
    
    # Try to run actual coverage
    run_pytest_coverage()
    
    # Final summary
    print("\n🎯 SUMMARY:")
    print("=" * 60)
    print(f"📈 Current Estimated Coverage: {current_coverage:.1f}%")
    print(f"🎯 Target Coverage: 90.0%")
    print(f"📊 Gap to Close: {90.0 - current_coverage:.1f}%")
    print(f"📝 Tests to Create: {len(needed_tests)}")
    print(f"🔥 High Priority Actions: {len([r for r in recommendations if r['priority'] == 'HIGH'])}")
    
    if current_coverage >= 90:
        print("🎉 CONGRATULATIONS! You've reached 90% coverage!")
    elif current_coverage >= 70:
        print("🚀 Great progress! You're close to the 90% target!")
    elif current_coverage >= 50:
        print("📈 Good start! Keep building comprehensive tests!")
    else:
        print("🏗️ Building foundation! Focus on core component tests first!")


if __name__ == "__main__":
    main()