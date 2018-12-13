"""
setup.py file for SWIG
"""

from distutils.core import setup, Extension
import os

# Get pythia path from env
try:
    path_to_pythia = os.environ['PYTHIA_PATH']
    print "Pythia_path:", path_to_pythia
except KeyError:
    print 'In setup.py: could not PYTHIA_PATH, check the paths in compile.sh'
    exit(1)

if not path_to_pythia.endswith('/'):
    path_to_pythia += '/'

analysis_module = Extension('_Analysis',
                            sources=['analysis_wrap.cxx', 'analysis.cpp'],
                            include_dirs=[path_to_pythia + 'include'],
                            library_dirs=[path_to_pythia + 'lib'],
                            libraries=['pythia8'],
                            extra_compile_args = ['-std=c++11', '-O2', '-pedantic', '-Wno-shorten-64-to-32']
                            )

setup (name = 'Analysis',
       version = '0.1',
       author      = "S/I",
       description = """Analysis module""",
       ext_modules = [analysis_module],
       py_modules = ["Analysis"],
       )
