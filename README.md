# ML-2DHM

This repository contains the code used to produce results in the paper *Signal
mixture estimation for degenerate heavy Higgses using a deep neural network*
published in EPJC: [link (open access)](https://link.springer.com/article/10.1140%2Fepjc%2Fs10052-018-6455-z?fbclid=IwAR1PmOAVx7xrnDU71vuepJEvN5w_nXAXWA0YEFc0Ym6fgY8i_qH3h_8WUnY).

```
@Article{Kvellestad2018,
    author="Kvellestad, Anders
    and Maeland, Steffen
    and Str{\"u}mke, Inga",
    title="Signal mixture estimation for degenerate heavy Higgses using a deep neural network",
    journal="The European Physical Journal C",
    year="2018",
    month="Dec",
    day="12",
    volume="78",
    number="12",
    pages="1010",
    issn="1434-6052",
    doi="10.1140/epjc/s10052-018-6455-z",
    url="https://doi.org/10.1140/epjc/s10052-018-6455-z"
}
```

### Directory structure

- `pythia`: Contains code for generating Monte Carlo events and performing
    analysis.
- `generate_samples`: Scripts to facilitate event generation for specific
    THDM parameter. Creates train/test/validation data sets.
- `plot`: Scripts to plot feature distributions
- `neuralnet`: Scripts to train neural networks
- `measure_yields`: Create templates for likelihood fits, make network
    predictions on testing data and assess performance
- `scan`: Scan over THDM model parameters and calculate cross sections times
    branching ratios

Each directory has a README file explaining how to run the code. 

### Dependencies

The scripts are written for Python 2.7 and require
- Keras
- Tensorflow
- Numpy
- Scipy
- Scikit-learn
- h5py
- Matplotlib

In addition, the following physics software is needed:
- [Pythia8.2](http://home.thep.lu.se/~torbjorn/pythia81html/Welcome.html),
    version 8.219
- [2HDMC](https://2hdmc.hepforge.org), version 1.7.0
- [HiggsBounds and HiggsSignals](https://higgsbounds.hepforge.org), versions
    4.3.1 and 1.4.0 respectively
- [SusHi](https://sushi.hepforge.org), version 1.6.1
- [ROOT](https://root.cern.ch), version 6.10.00

The version numbers are the ones used for the paper; other versions might work
too.


### Reproduce the results

The following steps reproduces the results and plots shown in the paper.

1. Download and compile the programs listed above.
2. Edit [`env.sh`](env.sh) to point to the correct installation paths.
3. Do `source env.sh` 
4. Go to the [`pythia`](pythia) directory, and follow instructions for compiling the
    analysis code against the Pythia installation.
5. Go to the [`generate_samples`](generate_samples) directory, and follow instructions to generate
    simulated events.
6. Go to the [`neuralnet`](neuralnet) directory, and follow instructions to train a neural
    network.
7. Go to the [`measure_yields`](measure_yields) directory, and follow instructions to create
    templates, run fits for the phistar and neural net methods, and obtain
    results.

