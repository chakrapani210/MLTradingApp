import os

def count_lines_in_src():
    src_files = []
    total_lines = 0
    
    for root, dirs, files in os.walk('src'):
        # Skip __pycache__ directories
        if '__pycache__' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                src_files.append(file_path)
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                        total_lines += lines
                        print(f'{file_path}: {lines} lines')
                except Exception as e:
                    print(f'Error reading {file_path}: {e}')
    
    print(f'\nSUMMARY:')
    print(f'Total Python files in src/: {len(src_files)}')
    print(f'Total lines of code: {total_lines}')
    print(f'Average lines per file: {total_lines / len(src_files) if src_files else 0:.1f}')
    
if __name__ == '__main__':
    count_lines_in_src()