
# External program paths:
# THESE ARE TO BE SET BY THE USER.

# Set path top Pythia install directory
export PYTHIA_PATH=/path/to/pythia8219
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PYTHIA_PATH/lib

# Set path to ROOT install directory, then set up
ROOTPATH=/path/to/root_v6.10.00
source $ROOTPATH/bin/thisroot.sh

# Set path to SusHi binary
export SUSHIPATH=/path/to/SusHi-1.6.1/bin/sushi

# Path to 2HDMC install directory
export TWOHDMCPATH=/path/to/2HDMC-1.7.0

# Path to HiggsBounds install directory
export HIGGSBOUNDSPATH=/path/to/HiggsBounds-4.3.1

# Path to HiggsSignals install directory
export HIGGSSIGNALSPATH=/path/to/HiggsSignals-1.4.0



# Internal paths
# Do not modify.

# Make our python modules available
export PYTHONPATH=$(pwd):$PYTHONPATH

# Set path to our 2HDMC binary
export TWOHDMCCODEPATH=$(pwd)/generate_samples/2hdmc
