# PPA Parametrization

This repo holds the code for running a number of experiments for the parametrization of the plant propagation algorithm.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Prerequisite libraries, frameworks, and modules can be installed through:

```
pip install -r requirements.txt
```

This will install the correct versions of:
- matplotlib (3.0.3)
- pandas (0.24.2)
- scipy (1.1.0)
- numpy (1.15.4)
- seaborn (0.9.0)

### Repository
The following list describes the most important files in the project. Parts of the code originate from an existing research on PPA that is done before, of which the documentation can be found here: 
https://github.com/WouterVrielink/FWAPPA.
- **/Code**: contains all the code of this project. 
  - **/Code/batchrunner.py**: contains the code to run all experiments. (Author: W. L. J. Vrielink)
  - **/Code/benchmark_functions.py**: contains all the benchmark functions. (Author: W. L. J. Vrielink)
  - **/Code/benchmarks.py**: contains the Benchmark class, the set_benchmark_properties decorator, 
  the param_shift helper function, and the apply_add function. (Author: W. L. J. Vrielink)
  - **/Code/configuration_runner.py**: contains a number of statistical options to calculate over the range of tested parameter configurations.
  - **/Code/environment.py**: contains the Environment class. (Author: W. L. J. Vrielink)
  - **/Code/heatmap.py**: contains the Heatmap class. Running this file results in a number of heatmap visualizations
  for multiple benchmark test functions.
  - **/Code/helpers.py**: contains the code to create a number of parameter configurations for the specified range for the parameters popSize and n_max, as well as necessary helper functions for the heatmap visualizations. 
  - **/Code/plantpropagation.py**: contains the code of the Plant Propagation algorithm. (Author: W. L. J. Vrielink)
  - **/Code/point.py**: contains a coordinate class that can calculate distances and determine if
    it was already once evaluated. (Author: W. L. J. Vrielink)
    
## Contributing

Please read [CONTRIBUTING.md](https://github.com/WouterVrielink/FWAPPA/blob/master/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **M. R. H. de Jonge**
* **W. L. J. Vrielink**
