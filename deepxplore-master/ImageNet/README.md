# DeepXplore - ImageNet

Run with below commands; we did follow original paper for the parameters.
```
python gen_diff.py occl 1 0.1 10 100 20 0.2 5
python gen_diff.py light 1 0.1 10 100 20 0.2 5
python gen_diff.py occl 1 0.1 10 100 20 0.2 5 --param multi
python gen_diff.py light 1 0.1 10 100 20 0.2 5 --param multi
python gen_diff.py occl 1 0.1 10 75 20 0.2 5 --param strong
python gen_diff.py light 1 0.1 10 75 20 0.2 5 --param strong
python gen_diff.py occl 1 0.1 10 25 20 0.2 5 --param boundary
python gen_diff.py light 1 0.1 10 25 20 0.2 5 --param boundary
```
Above parameters are `transformation`(blackout, light, occl), `weight_diff`, `weight_nc`, `step`, `minmax_seeds`, `seeds`, `grad_iterations`, `threshold`, `k`(for k-multisection), `coverage_type`(named param).  

### NC
- NC1: Basic Neuron Coverage
- NC2: k-multisection Coverage
- NC3: (Strong-)Boundary Neuron Coverage

### Notes
- transformation - blackout did not work for ImageNet. You can also look at [the original repo](https://github.com/peikexin9/deepxplore/tree/master/ImageNet/generated_inputs) - it does not contain blackout_* in generated inputs.