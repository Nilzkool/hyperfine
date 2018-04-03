# -*- coding: utf-8 -*-
"""
This is a wrapper module to search for optimal hyperparameters using tree of parzen estimators (tpe)
or random search and then to fine tune the parameters using grid search. tpe and random search are 
implemented using hyperopt library (Bergstra, James S., et al)  while the grid search is written 
by myself.

"""

import numpy as np
from hyperopt import fmin, tpe, rand,hp, Trials

class search_routine:
    
    def __init__(self,obj_func, space, algo, max_evals, verbose):
        self.obj_func=obj_func
        self.algo=algo
        self.space=space
        self.max_evals=max_evals
        self.verbose=verbose
        self.hyperopt_space=None
        
    def __prepare_grid(self,current_opt_params):
        #Function to prepare the grid of parameters
       
        grid_points_list=[]
        i_cnt=0
        grid_size_ind=0
        grid_size_flag=False
        param_names=[] 
        
        # Calculate 1D points
        for param in self.space:  
            param_details=self.space[param]
            
            fine_tune_flag=param_details[3]
            sampling=param_details[1]
            opt_param=current_opt_params[param]
            
            if fine_tune_flag==False or sampling=='choice':
                oneD_points=np.array([opt_param])
            else:
                grid_params=param_details[3]
                radius=(grid_params[0])/100
                N_points=grid_params[1]
                if (sampling=='quniform') or (sampling=='qloguniform') or (sampling=='randint') :
                    lb=np.ceil(opt_param-radius*opt_param)
                    lb=max(1,lb)
                    ub=np.floor(opt_param+radius*opt_param)
                    if lb==ub:
                        ub=ub+1e-4 # handling randint break down when lb==ub
                    oneD_points=np.linspace(lb,ub,N_points)
                    oneD_points.astype(int)
                    
                else:
                    lb=opt_param-radius*opt_param
                    ub=opt_param+radius*opt_param
                
                    oneD_points=np.linspace(lb,ub,N_points)
                if grid_size_flag==False:    
                    grid_size_flag=True
                    grid_size_ind=i_cnt+1
                
            grid_points_list.append(oneD_points)
            param_names.append(param)    
            i_cnt+=1  
        
        # Create grid
        grid = np.meshgrid(*[points for points in grid_points_list])
        
        # Flatten grid
        num_params=len(self.space)
        grid_size=grid[grid_size_ind].size
        
        flattened_grid=np.zeros((grid_size,num_params))
        
        i_cnt=0
        for uni_grid in grid:
            uni_grid_flat=uni_grid.flatten()
            flattened_grid[:,i_cnt]=uni_grid_flat
            i_cnt+=1
        
        return flattened_grid, param_names     


    def __construct_grid_param_dict(self,param_vals,param_names):
        # Function to construct a parameter dict for obj function from grid values
        
        param_dict={}
        
        i_cnt=0
        for param in param_names:
            param_details=self.space[param]
            sampling=param_details[1]
            
            if sampling=='choice':
                option=int(param_vals[i_cnt])
                choices=param_details[2]
                option_str=choices[option]
                param_dict[param]=option_str
            
            else:
                param_dict[param]=param_vals[i_cnt]
                
            i_cnt+=1
        
        return param_dict
                
    
    def __grid_fine_tune(self,final_dict):
        # Function for grid tuning
        
        print("Initiate grid fine tuning...")
        min_cost=final_dict['results']['loss']
        
        cur_opt_params=final_dict['best_params']
        flat_grid, param_names=self.__prepare_grid(cur_opt_params)
        
        n_evaluations=flat_grid.shape[0]
        for i_eval in range(n_evaluations): 
            param_vals=list(flat_grid[i_eval,:])
            param_dict=self.__construct_grid_param_dict(param_vals,param_names)
            returned_dict= self.obj_func(param_dict)
            cost=returned_dict['loss']
            if cost<min_cost:
                min_cost=cost
                final_dict={'results':returned_dict}
                final_dict['best_params']=param_dict
                
            if self.verbose==1:
                print('Fine tune evaluation no '+ str(i_eval+1)+'/'+str(n_evaluations)+ ': Current cost = ', cost, ', Minimum cost = ',min_cost)     
           
        return final_dict
    
    def __construct_hyperopt_space(self):
        # Function to contruct the space for fmin in hyperopt 
        
        hyperopt_space={}
        for param in self.space:
            param_details=self.space[param]
            label=param_details[0]
            sampling_type=param_details[1]
            bounds=param_details[2]
            if sampling_type=='choice':
                hyperopt_space[param]=hp.choice(label,bounds)
            elif sampling_type=='uniform':
                hyperopt_space[param]=hp.uniform(label,bounds[0], bounds[1])
            elif sampling_type=='randint':
                hyperopt_space[param]=hp.randint(label,bounds[0])
            elif sampling_type=='quniform':
                hyperopt_space[param]=hp.quniform(label,bounds[0], bounds[1], bounds[2])
            
            elif sampling_type=='loguniform':
                hyperopt_space[param]=hp.loguniform(label,bounds[0], bounds[1])
            
            elif sampling_type=='qloguniform':
                hyperopt_space[param]=hp.qloguniform(label,bounds[0], bounds[1])
        
        return hyperopt_space
        
    def __convert_choices2str(self,final_dict):
        # Function to convert choice ints to choice strs
        best_params=final_dict['best_params']
        for param in best_params:
            param_value=best_params[param]
            param_space=self.space[param]
            if param_space[1]=='choice' and isinstance(param_value, str)==False:
                param_value_bounds=param_space[2]
                best_params[param]=param_value_bounds[int(param_value)]
        
        return final_dict   
     
    def search(self):
        # Main callabale fubction
        
        algo=self.algo.split('+')
        main_algo=algo[0]
        
        if len(algo)>1:
            fine_algo=algo[1]
        else:
            fine_algo=None
        
        algo_dict={'tpe': tpe.suggest,'random':rand.suggest}
        
        if main_algo not in algo_dict:
            print('Please enter either tpe or random as the main algorithm')
            return False
        
        print("Hyperparameter search initiated...")
        
        self.hyperopt_space=self.__construct_hyperopt_space()
        trials=Trials()
        best_params = fmin(self.obj_func, self.hyperopt_space, algo=algo_dict[main_algo], max_evals=self.max_evals, trials=trials,verbose=self.verbose)
        best_trials_dict=trials.best_trial
        
        print("Hyperparameter search over...")
        
            
        final_dict={'best_params': best_params, 'results': best_trials_dict['result']}
        
        if fine_algo=="grid":
            final_dict=self.__grid_fine_tune(final_dict)
            
        else:
            final_dict={'best_params': best_params, 'results': best_trials_dict['result']}
        
        # convert to choice strings
        final_dict=self.__convert_choices2str(final_dict)
   
        
        return final_dict