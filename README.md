# hyperfine
hyperfine is a wrapper for hyperparameter optimization using Tree of Parzen Estimators (TPE) and random search that further allows fine tuning of parameters using grid search. TPE and random search are implemented using the hyperopt library (Bergstra et al., https://github.com/hyperopt/hyperopt) while the grid search was written by me. The reason I wrote the grid search from scratch is because this implementation allows I wanted the following flexibilities:
- 
