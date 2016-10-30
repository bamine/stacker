# stacker
Stacker is a library for automating repetitive tasks in a Machine Learning process.
It is specially thought for competitive Machine Learning.
# Getting Started
## Installing package
To install the package (as well as the requirements)
```
pip install git+https://github.com/bamine/stacker.git
```
## Optimization
Let's start with a toy classification task
```python
In [1]: from sklearn import datasets
In [2]: X, y = datasets.make_classification()
```
Next we need to define out`Task` instance, that describes the problem at hand and the evaluation method
```python
In [3]: from stacker.optimization.task import Task
In [4]: task = Task(name="my_test_problem", X=X, y=y, task="classification", test_size=0.1, random_state=42)
```
We also need to define a scorer to evaluate the performance of our models, it will represent the function to minimize in our optimization
```python
In [5]: from stacker.optimization.scorer import Scorer
In [6]: from sklearn.metrics import roc_auc_score
In [7]: scorer = Scorer(name="auc_error", scoring_function=lambda y_pred, y_true: 1 - roc_auc_score(y_pred, y_true))
```
That's it, that's all we need to start our optimization
```python
In [8]: from stacker.optimization.optimizers import XGBoostOptimizer
In [9]: optimizer = XGBoostOptimizer(task=task, scorer=scorer)
In [10]: best = optimizer.start_optimization(max_evals=10)
```
We see optimization logs printing to the screen:
```
2016-10-30 20:06:58,908 optimization INFO     Started optimization for task Task=my_test_problem - type=classification - validation_method=train_test_split - len(X)=100
2016-10-30 20:06:58,940 hyperopt.tpe INFO     tpe_transform took 0.010630 seconds
2016-10-30 20:06:58,940 hyperopt.tpe INFO     TPE using 0 trials
2016-10-30 20:06:58,944 optimization INFO     Evaluating with test size 0.1 with parameters {'base_score': 0.15131750274218725, 'n_estimators': 1549, 'max_delta_step': 5.456151637154244, 'max_depth': 32, 'scale_pos_weight': 17.01641396909963, 'subsample': 0.5602243542512042, 'min_child_weight': 7.357276709287346, 'reg_lambda': 0.04892385490528424, 'colsample_bylevel': 0.9270255573832653, 'reg_alpha': 9.241575982835831, 'learning_rate': 0.334, 'gamma': 9.204287327296093, 'colsample_bytree': 0.7401493524610046}
2016-10-30 20:06:58,944 optimization INFO     Training model ...
2016-10-30 20:06:59,445 optimization INFO     Training model done !
2016-10-30 20:06:59,449 optimization INFO     Score = 0.5
```
After `max_evals` round of optimization, the variable `best` holds the best parameters for our model so far:
```python
In [11]: best
Out[11]:
{'base_score': 0.11896730519023568,
 'colsample_bylevel': 0.7784020826729505,
 'colsample_bytree': 0.5406725444519914,
 'gamma': 2.8852788849467914,
 'learning_rate': 0.0458,
 'max_delta_step': 7.94440873152742,
 'max_depth': 40.54677208316948,
 'min_child_weight': 1.4917791224018282,
 'n_estimators': 4098.139372543198,
 'reg_alpha': 5.145060179230331,
 'reg_lambda': 4.275665709200242,
 'scale_pos_weight': 92.51593088136909,
 'subsample': 0.30804272389187076}
```
To see the performance of our best parameters:
```python
In [16]: optimizer.score(best)
2016-10-30 20:09:58,918 optimization INFO     Evaluating with test size 0.1 with parameters {'base_score': 0.11896730519023568, 'n_estimators': 4098, 'max_delta_step': 7.94440873152742, 'max_depth': 40, 'scale_pos_weight': 92.51593088136909, 'subsample': 0.30804272389187076, 'min_child_weight': 1.4917791224018282, 'reg_lambda': 4.275665709200242, 'colsample_bylevel': 0.7784020826729505, 'reg_alpha': 5.145060179230331, 'learning_rate': 0.0458, 'gamma': 2.8852788849467914, 'colsample_bytree': 0.5406725444519914}
2016-10-30 20:09:58,918 optimization INFO     Training model ...
2016-10-30 20:10:00,227 optimization INFO     Training model done !
2016-10-30 20:10:00,228 optimization INFO     Score = 0.0
Out[16]: {'loss': 0.0, 'status': 'ok'}
```