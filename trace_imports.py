#!/usr/bin/env python3
"""
Import Tracer - Find files actually used when running main.py
"""

import sys
import os
import importlib.util
from pathlib import Path

def trace_imports():
    """Trace which files are actually imported by main.py"""
    project_root = os.getcwd()
    imported_files = set()
    
    # Track all .py files in the project
    all_py_files = set()
    for root, dirs, files in os.walk(project_root):
        # Skip .venv and __pycache__
        if '.venv' in root or '__pycache__' in root or '.git' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(root, file)
                all_py_files.add(os.path.relpath(full_path, project_root))
    
    # Hook into import system
    original_import = __builtins__.__import__
    
    def import_hook(name, globals=None, locals=None, fromlist=(), level=0):
        try:
            module = original_import(name, globals, locals, fromlist, level)
            
            # Track files from our project
            if hasattr(module, '__file__') and module.__file__:
                file_path = os.path.relpath(module.__file__, project_root)
                if not file_path.startswith('..') and not 'site-packages' in file_path:
                    imported_files.add(file_path.replace('\\', '/'))
            
            return module
        except:
            return original_import(name, globals, locals, fromlist, level)
    
    # Replace import function
    __builtins__.__import__ = import_hook
    
    try:
        # Import main.py (this will trigger all its imports)
        import main
        
        # Try to create TradingSystemMain to see what else gets imported
        trading_system = main.TradingSystemMain()
        
    except Exception as e:
        print(f"Error during import tracing: {e}")
    finally:
        # Restore original import
        __builtins__.__import__ = original_import
    
    return imported_files, all_py_files

if __name__ == "__main__":
    print("🔍 Tracing imports from main.py...")
    
    imported_files, all_py_files = trace_imports()
    
    print(f"\n✅ FILES ACTUALLY USED ({len(imported_files)}):")
    for file in sorted(imported_files):
        print(f"  ✓ {file}")
    
    unused_files = all_py_files - imported_files
    
    print(f"\n❌ POTENTIALLY UNUSED FILES ({len(unused_files)}):")
    for file in sorted(unused_files):
        # Skip some obvious files that might be legitimately unused
        if any(skip in file for skip in ['test_', '__pycache__', '.venv', 'backup_files', 
                                        'simple_test_runner.py', 'count_src_lines.py',
                                        'create_architecture_diagram.py']):
            continue
        print(f"  ✗ {file}")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Total Python files: {len(all_py_files)}")
    print(f"  Actually imported: {len(imported_files)}")
    print(f"  Potentially unused: {len(unused_files)}")