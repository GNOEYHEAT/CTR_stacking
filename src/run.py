import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, TargetEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from config import getConfig
from model import xgboost, lightgbm, catboost
# import wandb
# wandb.init(project="DACON_236258", name="stack")

args,parser = getConfig()

# wandb.config.update(args)

scaler = args.scaler
cv = args.cv
seed = args.seed

if scaler == "standard":
    scaler = StandardScaler()
elif scaler == "minmax":
    scaler = MinMaxScaler()
elif scaler == "robust":
    scaler = RobustScaler()

def set_seeds(seed=seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seeds()

# submission_id = f"{parser.description}"
submission_id = f"{parser.description}"
submission_id

train_df = pd.read_parquet(f'../data/pp_train_ce.parquet', engine='pyarrow')
test_df = pd.read_parquet(f'../data/pp_test_ce.parquet', engine='pyarrow')

# train_df.shape, test_df.shape

train_x = train_df.drop('Click', axis=1)
train_y = train_df['Click']

test_x = test_df.copy()

# train_x.shape, test_x.shape

def add_mean_by_feature(train,test, feature_name, numeric_columns):
    for nfeat in numeric_columns:
        df_by_feature = train.groupby(feature_name)[nfeat].mean().rename("mean_by_" + feature_name+'_'+nfeat)
        train = train.merge(df_by_feature, on=feature_name, how="left")
        test = test.merge(df_by_feature, on=feature_name, how="left")
    return train,test

for feature in ['F21','F09','F37','F02']:
    train_x, test_x = add_mean_by_feature(
        train_x, test_x, feature, ['F32','F24','F04','F29','F11','F06']
    )

# train_x.shape, test_x.shape


def get_stacking_ml_datasets(model, X_train_n, y_train_n, X_test_n, n_folds, fitting=True):
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    train_fold_pred = np.zeros((X_train_n.shape[0], 1))
    test_pred = np.zeros((X_test_n.shape[0], n_folds))
    
    for folder_counter, (train_index, valid_index) in enumerate(skf.split(X_train_n, y_train_n)):
        X_tr = X_train_n[train_index]
        y_tr = y_train_n[train_index]
        X_te = X_train_n[valid_index]
        
        if fitting == True:
            model.fit(X_tr, y_tr)
            
        train_fold_pred[valid_index, :] = model.predict_proba(X_te)[:, 1].reshape(-1, 1)
        test_pred[:, folder_counter] = model.predict_proba(X_test_n)[:, 1]
        
    test_pred_mean = np.mean(test_pred, axis=1).reshape(-1, 1)    
    
    return train_fold_pred, test_pred_mean

# model
best_ml = [
    xgboost,
    lightgbm,
    catboost,
]

print('best_ml',best_ml)

X_train = train_x.copy()
y_train = train_y.copy()
X_test = test_x.copy()

ohe_col = ['F15']

train_ohe = []
test_ohe = []
for i in ohe_col:
    ohe = OneHotEncoder(handle_unknown="ignore")
    ohe = ohe.fit(X_train[i].values.reshape(-1, 1))
    train_ohe.append(ohe.transform(X_train[i].values.reshape(-1, 1)).toarray())
    test_ohe.append(ohe.transform(X_test[i].values.reshape(-1, 1)).toarray())
train_ohe = np.hstack(train_ohe)
test_ohe = np.hstack(test_ohe)

X_train = X_train.drop(ohe_col, axis=1)
X_test = X_test.drop(ohe_col, axis=1)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = np.concatenate((X_train, train_ohe), axis=1)
X_test = np.concatenate((X_test, test_ohe), axis=1)

# X_train.shape, X_test.shape

# %%time

meta_ml_X_train=[]
meta_ml_X_test=[]

for estimator in best_ml:

    print(estimator)
    
    temp_X_train, temp_X_test = get_stacking_ml_datasets(estimator, X_train, y_train.values, X_test, cv)
    
    meta_ml_X_train.append(temp_X_train)
    meta_ml_X_test.append(temp_X_test)
    
meta_ml_X_train = np.hstack(meta_ml_X_train)
meta_ml_X_test = np.hstack(meta_ml_X_test)

# meta_ml_X_train.shape, meta_ml_X_test.shape

meta_clf = LogisticRegression(n_jobs=-1, random_state=seed)

meta_clf.fit(meta_ml_X_train, y_train)
prediction = meta_clf.predict_proba(meta_ml_X_test)

# prediction.shape

# submission
submission = pd.read_csv("../data/sample_submission.csv")
submission["Click"] = prediction[:, 1]
submission.to_csv(f"submission/{submission_id}.csv", index=False)

# submission.head()
# submission["Click"].apply(lambda x : 1 if x>0.5 else 0).value_counts(normalize=True)