
## Prerequisites:
A trained neural network. If no names have been changed, it runs as-is,
otherwise adjust the file names in [`compare_methods_on_different_thetas.py`](compare_methods_on_different_thetas.py).

## Predict on single test set
To reproduce figures 2 and 3, run

´´´
cd results
python compare_methods_on_different_thetas.py --test --recreate
´´´

The phistar template creation and maximum likelihood fits are implemented
in [`phistar_predictor.py`](phistar_predictor.py), while the neural network
method is implemented in [`strumkes_predictor.py`](strumkes_predictor.py).
ML fits are performed using RooFit and Minuit2.


## Predict for multiple test sets, calculate prediction uncertainty
To reproduce figures 4, 5 and 6, run the following two steps:
```
python compare_methods_on_different_thetas.py --validation
python compare_methods_on_different_thetas.py --plot 
```


