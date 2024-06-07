
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score
from config import getConfig

args,parser = getConfig()
seed = args.seed

xgboost_params = {
    'n_estimators': 3943,
    'max_depth': 16, 
    'learning_rate': 0.014877207226997192,
    'gamma': 5.319373833645427e-06, 
    'min_child_weight': 35, 
    'subsample': 1.0, 
    'sampling_method': 'uniform', 
    'colsample_bytree': 0.5, 
    'reg_alpha': 3.154080705017364e-08, 
    'reg_lambda': 0.0007278615348762349,
    'tree_method': 'gpu_hist',
    'n_jobs' : -1,
    'random_state': seed,
    'eval_metric' : roc_auc_score,
} # 0.7848833943435063 

xgboost = XGBClassifier(**xgboost_params)

lightgbm_params = {
    'data_sample_strategy': 'bagging',
    'learning_rate': 0.08075877732474707,
    'n_estimators': 3903,
    'min_child_weight': 9.345466358862428, 
    'subsample': 0.6, 
    'colsample_bytree': 0.8, 
    'reg_alpha': 2.9790490549549577e-06, 
    'reg_lambda': 0.005072096330288197,
    'verbosity': -1,
    'device_type': 'gpu',
    'random_state': seed,
    'n_jobs' : -1,
    'metric' : 'auc',
} # 0.7737584662106686  

lightgbm = LGBMClassifier(**lightgbm_params)

catboost_params = {
    'iterations': 3735,
    'learning_rate': 0.08669080239545442,
    'depth': 12,
    'l2_leaf_reg': 0.5649366200485388,
    'loss_function': 'CrossEntropy', 
    'od_pval': 0.01,
    'random_seed': seed,
    'verbose': 0,
    'random_strength': 0.7,
    'task_type': 'GPU',
    'boosting_type': 'Ordered',
} # 0.7674564182253708

catboost = CatBoostClassifier(**catboost_params)
