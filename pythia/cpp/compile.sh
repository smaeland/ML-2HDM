# Script to compile the analysis module

# Re-running SWIG not necessary unless cpp/h files are changed
# Can be left empty. 
path_to_swig=""


if [[ -z "$PYTHIA_PATH" ]]; then
    echo "PYTHIA_PATH not set - cannot continue"
    exit 1
fi

# If SWIG is available, we can regenerate the wrapper files
if [[ -n "$path_to_swig" ]]; then
    echo "Regenerating wrappers..."
    eval "${path_to_swig}/swig -python -c++ -I${PYTHIA_PATH}/include analysis.i"
    if [[ $? -ne 0 ]]; then
        echo "Swig failed! Using existing wrapper files"
    else
        echo "OK."
    fi
else
    echo "Using existing wrapper files"
fi

# Compile python library
echo "Compiling python library"

# First confirm that pythia is in LD_LIBRARY_PATH
if ! [[ $LD_LIBRARY_PATH =~ .*pythia.* ]]; then
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${PYTHIA_PATH}/lib
fi

export ARCHFLAGS='-arch x86_64'

eval "python setup.py build_ext --inplace"
if [[ $? -ne 0 ]]; then
    echo "Build failed"
    exit 2
else
    echo "OK."
fi

