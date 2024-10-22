import os
import sys
from sage.all import preparse

def preprocess_sage_files(source_dir):
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            if filename.endswith('.sage'):
                sage_file = os.path.join(root, filename)
                py_file = os.path.join(root, filename[:-5] + '.py')
                print(f"Preprocessing {sage_file} -> {py_file}")
                with open(sage_file, 'r') as sf:
                    content = sf.read()
                preprocessed_content = preparse(content)
                with open(py_file, 'w') as pf:
                    pf.write(preprocessed_content)

if __name__ == '__main__':
    source_directory = sys.argv[1] if len(sys.argv) > 1 else 'src/pyPlumbing'
    preprocess_sage_files(source_directory)

