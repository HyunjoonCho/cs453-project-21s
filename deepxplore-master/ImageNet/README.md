# DeepXplore - ImageNet

Run with below commands; we did follow original paper for the parameters.
```
python gen_diff.py basic occl 1 0.1 10 100 20 0.2
python gen_diff.py basic light 1 0.1 10 100 20 0.2
python gen_diff.py multi occl 1 0.1 10 100 20 0.2 -k 5
python gen_diff.py multi light 1 0.1 10 100 20 0.2 -k 5
python gen_diff.py strong occl 1 0.1 10 100 20 0.2
python gen_diff.py strong light 1 0.1 10 100 20 0.2
python gen_diff.py boundary occl 1 0.1 10 100 20 0.2
python gen_diff.py boundary light 1 0.1 10 100 20 0.2
```
Above parameters are `transformation`(blackout, light, occl), `weight_diff`, `weight_nc`, `step`, `minmax_seeds`, `seeds`, `grad_iterations`, `threshold`, `k`(for k-multisection), `coverage_type`(named param).  

### NC
- NC1: Basic Neuron Coverage
- NC2: k-multisection Coverage
- NC3: (Strong-)Boundary Neuron Coverage

### Notes
- transformation - blackout did not work for ImageNet. You can also look at [the original repo](https://github.com/peikexin9/deepxplore/tree/master/ImageNet/generated_inputs) - it does not contain blackout_* in generated inputs.