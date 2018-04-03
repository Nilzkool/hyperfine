# hyperfine
hyperfine is a wrapper for hyperparameter optimization using Tree of Parzen Estimators (TPE) and random search that further allows fine tuning of parameters using grid search. TPE and random search are implemented using the hyperopt library (Bergstra et al., https://github.com/hyperopt/hyperopt) while the grid search was written by me. The reason I wrote the grid search from scratch was because  I wanted the following flexibilities:
1. Handling dictionary returns containing multiple pieces of information as implemented in hyperopt
2. The option to indicate which hyper-parameter needs to be fine tuned
3. Specifying fine tune settings for individual parameters

In the current version, parallel evaluation is not supported.


