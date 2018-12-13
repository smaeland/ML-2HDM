
## Prerequisites:
Ensure that Keras with Tensorflow is installed and working. 


## Train the neural network model:
In `train/`, run
```
python train_networks.py --input [path to training set]
```
e.g.
```
python train_networks.py --input ../../generate_samples/450GeV/train/
```

Training on GPUs is supported, and enabled by the `--gpu` flag.


