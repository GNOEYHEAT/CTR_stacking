import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

from pyarrow import csv

import plotly.express as px
import category_encoders as ce

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, TargetEncoder 

from tqdm import tqdm

import argparse

# import wandb
# wandb.init(project="DACON_236258", name="preprocess")

parser = argparse.ArgumentParser(description="preprocess")
parser.add_argument('--seed', default=826, type=int)
args = parser.parse_args('')

# wandb.config.update(args)

seed = args.seed

def set_seeds(seed=seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

set_seeds()


train = csv.read_csv('../data/train.csv').to_pandas()
test = csv.read_csv('../data/test.csv').to_pandas()

train = train.drop(["ID"], axis=1)
test = test.drop(["ID"], axis=1)

encoding_num = list(train.dtypes[train.dtypes!="object"].index)

for col in encoding_num:
    if col != "Click":
        print(col)
        train[col] = train[col].fillna(0)
        test[col] = test[col].fillna(0)


encoding_target = list(train.dtypes[train.dtypes=="object"].index)

for col in encoding_target:
    print(col)
    train[col] = train[col].apply(lambda x : None if x=="" else x)
    test[col] = test[col].apply(lambda x : None if x=="" else x)

# Count Encoding

train_x = train.drop(['Click'], axis=1)
train_y = train['Click']
test_x = test.copy()

enc = ce.CountEncoder(cols=encoding_target).fit(train_x, train_y)
X_train_encoded = enc.transform(train_x)
X_test_encoded = enc.transform(test_x)

train_df = pd.concat([X_train_encoded, train_y], axis=1)
test_df = X_test_encoded.copy()

# train_df.shape, test_df.shape

train_df.to_parquet(f'../data/pp_train_ce.parquet', engine='pyarrow', index=False)
test_df.to_parquet(f'../data/pp_test_ce.parquet', engine='pyarrow', index=False)

# train_df.shape, test_df.shape