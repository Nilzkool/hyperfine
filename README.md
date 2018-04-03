# hyperfine
hyperfine is a wrapper for hyperparameter optimization using Tree of Parzen Estimators (TPE) and random search that further allows fine tuning of parameters using grid search. TPE and random search are implemented using the hyperopt library (Bergstra et al., https://github.com/hyperopt/hyperopt) while the grid search was written by me. The reason I wrote the grid search from scratch was because  I wanted the following flexibilities:
1. Handling dictionary returns containing multiple pieces of information as implemented in hyperopt
2. The option to indicate which hyper-parameter needs to be fine tuned
3. Specifying fine tune settings for individual parameters

## Requirements
- numpy
- hyperopt

## Example
This is a simple example to demonstrate on how to use this wrapper. Suppose we are interested in tuning a neural network model with one hidden layer and LBFGS solver. The parameters we need to tune are the number of hidden units, hidden activation type, the number of iterations and the l2 regularization which will be based on validation accuracy. Further, we also want to fine tune the number of iterations and l2 regularization parameters.

### Defining the objective functions
We start out by defining our model that will return specific information of the training and validation process.
```python
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

def train_and_evaluate(params):
    # read dataset
    global X, Y

    # unpack params
    hidden_units=int(params['hidden_units'])
    activation=params['activation']
    max_iter=int(params['iterations'])
    l2regu=params['l2_regu']
    
    # split into train and validation sets
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)
    
    # standerdize  data
    scaler.fit(X_train[:,:-1]) 
    X_train=scaler.transform(X_train)
    X_valid=scaler.transform(X_valid)
     
    # fit the neural net
    NNclf=MLPClassifier(hidden_layer_sizes=(hidden_units, ), activation=activation, solver='lbfgs',max_iter = max_iter,alpha=l2regu)
    NNclf.fit(X_train,Y_train) 
    
    # predictions
    pred_train=  NNclf.predict(X_train)
    pred_valid = NNclf.predict(X_valid)
     
    # evaluations
    train_accuracy=accuracy_score(Y_train,pred_train)
    train_sensitivity= recall_score(Y_train,pred_train)
    valid_accuracy=accuracy_score(Y_valid,pred_valid)
    valid_sensitivity=recall_score(Y_valid,pred_valid)
    
    return train_accuracy, train_sensitivity, valid_accuracy, valid_sensitivity
```

Now, we define the objective function.
```python
def cost_function(params):
    # train and evaluate the model and get the required information
    train_accuracy, train_sensitivity, valid_accuracy, valid_sensitivity = train_and_evaluate(params)  
    
    # prepare the return dictionary 
    cost_dict={'loss':-(valid_accuracy), 'status': 'ok',
                'Train accuracy': train_accuracy, 'Train sensitivity':train_sensitivity, 
                'Validation accuracy': valid_accuracy, 'Validation sensitivity': valid_sensitivity}
    return cost_dict
 ```
Please note that the __'loss':value__ and the __'status':'ok'__ entries of the cost dictionary are mandatory as they are required by the hyperopt package.

### Defining the search space
This is the most important part of the example which shows how to define the search space. The syntax is a bit different from defining the search space in hyperopt. In essence, every parameter key in the space dictionary should contain a list (will be referred to as 'param_space') of four values. These values are as follows:

#### param_space[0] = label 
A string identifier for pyll graphs in hyperopt)

#### param_space[1] = sampling_type 
One of the following strings: 'choice', 'uniform', 'quniform', 'loguniform' or 'qloguniform'. More details on these sampling types can be found (https://github.com/hyperopt/hyperopt/wiki/FMin) under section 2.1

#### param_space[2] =  bounds 
A list or tuple corresponding to the sampling type and shown as follows:
1. 'choice'- a list or tuple containing the desired options similar to options parameter in hp.choice
2. 'uniform' or 'loguniform'- a list or tuple formatted as (lower_bound, upper_bound)
3. 'quniform' or 'qloguniform'- a list or tuple formatted as (lower_bound, upper_bound, q)
4. 'randint'-a list or tuple formatted as (upper_bound)
Currently, the normal sampling types are not yet supported.

#### param_space[3] =  fine_tune_settings
This is a parameter that describes the fine_tune_settings. The value can either be False indicating fine tuning of the parameter is not required or a tuple containing (Radius, N_points). 
1. Radius - This is parameter expressed in percentage. After tpe or random search is carried out, a new search space is constructed around the current optimal parameters as follows: lower_bound= opt_param - opt_param x Radius/100 and upper_bound= opt_param + opt_param x Radius/100. 
2. N_points - Number of grid points required between the lower and upper bounds includeing the bounds.

So, in our example the space is coded as follows:
```python
space={'activation': ['activation','choice' ,['identity','logistic','tanh','relu'], False], 
       'hidden_units':['hidden_units', 'quniform', (1,10,1), False],
       'iterations': ['iterations', 'quniform', (50,250,1), (25,3)],
       'l2_regu': ['l2_regu', 'loguniform', (-4,0), (25,4)]
       }
```
Note that we do not require fine-tuning of activation and hidden_units parameters. By default, fine tuning of 'choice' parameters is ignored.

### Calling hyperfine module
Now, we will import the search_routine class from hyperfine wrapper. 
```python
import sys
folder_path='C:/folder_where_hyperfine_is_saved'
sys.path.insert(0,'')
from hyperfine import search_routine
```
search_routine class requires the following arguments:
1. obj_func- The objective or cost function
2. space-The search space
3. algo- One of the possible strings:'tpe+grid', 'random+grid', 'tpe' or 'random'. For instance, if we are interested in TPE optimization  and then fine tuning using grid search, the string should be 'tpe+grid'.
4. max_evals- Maximul number of evaluations
5. Verbose flag

In our example, we we can define it as follows:
```python
sr = search_routine(obj_func=Cost_Function, space=space, algo='tpe+grid', max_evals=1, verbose=1)
```
Finally, we invoke the search method which returns a dictionary containing the following keys:

1.'best_parameters' - A dictionary of optimal parameters that resulted in minimum loss
2. 'results' - A results dictionary containing all the information evaluated by the cost function at the optimal parameters
```python
best_param_dict=sr.search()
```

## Currently not supported
- parallel evaluation

## Further plans
- Add other fine tuning search methods
- Add parallel evaluations

## References
- Bergstra, James S., et al. “Algorithms for hyper-parameter optimization.” Advances in Neural Information Processing Systems. 2011.
- https://github.com/hyperopt/hyperopt/wiki/FMin
