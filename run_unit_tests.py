"""
Unit Test Runner
================
Comprehensive test runner for all unit tests in the tests/unit directory
"""

import sys
import os
import unittest
from pathlib import Path
import importlib.util


def discover_and_run_unit_tests():
    """Discover and run all unit tests in the tests/unit directory"""
    
    # Set up paths
    current_dir = Path(__file__).parent.absolute()
    unit_tests_dir = current_dir / 'tests' / 'unit'
    
    # Add project root to Python path
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    print("🔍 Unit Test Discovery and Execution")
    print("=" * 50)
    print(f"📁 Project root: {current_dir}")
    print(f"🧪 Unit tests directory: {unit_tests_dir}")
    
    if not unit_tests_dir.exists():
        print(f"❌ Unit tests directory not found: {unit_tests_dir}")
        return False
    
    # Discover all test files
    test_files = list(unit_tests_dir.glob('test_*.py'))
    
    if not test_files:
        print("❌ No test files found in unit tests directory")
        return False
    
    print(f"📄 Found {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  • {test_file.name}")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    total_tests = 0
    loaded_modules = 0
    
    # Load tests from each file
    for test_file in test_files:
        try:
            # Load the module
            spec = importlib.util.spec_from_file_location(
                f"test_module_{test_file.stem}", test_file
            )
            test_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(test_module)
            
            # Find test classes
            test_classes = []
            for name in dir(test_module):
                obj = getattr(test_module, name)
                if (isinstance(obj, type) and 
                    issubclass(obj, unittest.TestCase) and 
                    obj != unittest.TestCase):
                    test_classes.append(obj)
            
            # Add tests to suite
            for test_class in test_classes:
                tests = loader.loadTestsFromTestCase(test_class)
                suite.addTests(tests)
                total_tests += tests.countTestCases()
            
            loaded_modules += 1
            print(f"✅ Loaded {len(test_classes)} test classes from {test_file.name}")
            
        except Exception as e:
            print(f"❌ Failed to load tests from {test_file.name}: {e}")
            continue
    
    if total_tests == 0:
        print("❌ No test cases found")
        return False
    
    print(f"\n🎯 Total: {total_tests} test cases from {loaded_modules} modules")
    
    # Run the tests
    print("\n" + "="*60)
    print("🧪 RUNNING ALL UNIT TESTS")
    print("="*60)
    
    runner = unittest.TextTestRunner(
        verbosity=2,
        buffer=True,
        stream=sys.stdout
    )
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 UNIT TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100 if result.testsRun > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")
    
    # Show details of failures/errors
    if result.failures:
        print(f"\n❌ FAILURES ({len(result.failures)}):")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"{i}. {test}")
            print(f"   {traceback.splitlines()[-1] if traceback else 'No details'}")
    
    if result.errors:
        print(f"\n💥 ERRORS ({len(result.errors)}):")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"{i}. {test}")
            print(f"   {traceback.splitlines()[-1] if traceback else 'No details'}")
    
    # Final status
    tests_passed = len(result.failures) == 0 and len(result.errors) == 0
    
    print("\n" + "="*60)
    if tests_passed:
        print("🎉 ALL UNIT TESTS PASSED!")
        print("✅ Unit test suite is healthy")
    else:
        total_failed = len(result.failures) + len(result.errors)
        print(f"⚠️  {total_failed} UNIT TESTS FAILED")
        print("🔧 Please review and fix failing tests")
    
    print("="*60)
    
    return tests_passed


def run_specific_unit_test_file(test_file_name: str):
    """Run tests from a specific unit test file"""
    current_dir = Path(__file__).parent.absolute()
    unit_tests_dir = current_dir / 'tests' / 'unit'
    test_file = unit_tests_dir / test_file_name
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    # Add project root to Python path
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    print(f"🎯 Running specific unit test file: {test_file_name}")
    print("=" * 50)
    
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location("specific_test_module", test_file)
        test_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_module)
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Find and add test classes
        test_classes = []
        for name in dir(test_module):
            obj = getattr(test_module, name)
            if (isinstance(obj, type) and 
                issubclass(obj, unittest.TestCase) and 
                obj != unittest.TestCase):
                test_classes.append(obj)
        
        for test_class in test_classes:
            tests = loader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2, buffer=True)
        result = runner.run(suite)
        
        return len(result.failures) == 0 and len(result.errors) == 0
        
    except Exception as e:
        print(f"❌ Failed to run tests from {test_file_name}: {e}")
        return False


if __name__ == '__main__':
    # Check command line arguments
    if len(sys.argv) > 1:
        test_file_name = sys.argv[1]
        if not test_file_name.endswith('.py'):
            test_file_name += '.py'
        
        success = run_specific_unit_test_file(test_file_name)
    else:
        success = discover_and_run_unit_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)