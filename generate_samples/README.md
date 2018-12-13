# Generate training, testing, and validation data sets

## Prerequisites

If not done already, compile the analysis code as documented in the `pythia`
directory. 

Now compile the program that computes branching ratios:
```
cd 2hdmc
make
```

For cross section calculation, we use SusHi directly -- now further compilation
required as long as SusHi is installed.

## Generate events

The parameters of the model to generate events for are set in `generate.py`.
Only the masses are relevant to Pythia, while the remaining theory parameters
are required to compute the cross sections and branching ratios. The model
used for the paper is 
```
param_dict={'mh': 125, 'mH': 450, 'mA': 450, 'mC': 450, 'sin_ba': 0.999,
            'lambda_6': 0, 'lambda_7': 0, 'm12_2': 16000, 'tanb': 1}.
```
Other settings can be given to the `Model` instance, such as the luminosity (in
1/fb), and the name of the output directory.
The `GENERATION_SETTINGS` dict controls the number of events to generate, as well as
the switch for detector smearing. To create all three datasets, run
```
python generate.py --train --test --validation
```
Additional options are 
    - `--print_expected `: Print expected event yields
    - `--ncpu `: Set numbers of jobs to run simultaneously. Default 4.

