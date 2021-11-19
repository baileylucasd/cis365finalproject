import pandas as pd
import numpy as np
import datatable as dt
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc

import optuna
import tensorflow as tf

#%%time
train = pd.read_csv('../input/tabular-playground-series-oct-2021/train.csv')
test = pd.read_csv('../input/tabular-playground-series-oct-2021/test.csv')
sample_submission = pd.read_csv('../input/tabular-playground-series-oct-2021/sample_submission.csv')

memory_usage = train.memory_usage(deep=True) / 2 ** 11
start_memory = memory_usage.sum()


feature_cols = train.columns.tolist()[1:-1]
con_features = train.select_dtypes(include = 'float64').columns.tolist()
cat_features = train.select_dtypes(include = 'int64').columns.tolist()[1:-1]

train[con_features] = train[con_features].astype('float32')
train[cat_features] = train[cat_features].astype('uint8')

test[con_features] = test[con_features].astype('float32')
test[cat_features] = test[cat_features].astype('uint8')

memory_usage = train.memory_usage(deep=True) / 2 ** 11
end_memory = memory_usage.sum()

print('Memory usage decreased from {:.2f} MB to {:2f} MB ({:.2f} % reduction)'.format(start_memory, end_memory, 100 * (start_memory - end_memory) / start_memory))

from sklearn.preprocessing import StandardScaler

X = train.drop(columns=["id", "target"]).copy()
y = train["target"].copy()
X_test = test.drop(columns=["id"]).copy()

scaler = StandardScaler()
X = pd.DataFrame (data=scaler.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame (data=scaler.transform(X_test), columns=X_test.columns)


params = {
    'boosting_type':'dart', 
    'num_leaves': 31, 
    'max_depth': 8, 
    'n_estimators':1000, 
    'subsample_for_bin':2000, 
    'min_split_gain':0.0, 
    'min_child_weight':1e-3, 
    'min_child_samples':20, 
    'subsample':1.0, 
    'subsample_freq':0, 
    'colsample_bytree':1, 
    'reg_alpha':0, 
    'reg_lambda':0, 
}



try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
    BATCH_SIZE = tpu_strategy.num_replicas_in_sync * 64
    print("Running on TPU:", tpu.master())
    print(f"Batch Size: {BATCH_SIZE}")
    
except ValueError:
    strategy = tf.distribute.get_strategy()
    BATCH_SIZE = 512
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    print(f"Batch Size: {BATCH_SIZE}")

    with tpu_strategy.scope():
        kf = StratifiedKFold(n_splits=7, shuffle=True, random_state=786)

test_preds = []
scores = []

for fold, (idx_train, idx_valid) in enumerate(kf.split(X.iloc[:10000], y[:10000])):

    X_train, y_train = X.iloc[idx_train], y.iloc[idx_train]
    X_valid, y_valid = X.iloc[idx_valid], y.iloc[idx_valid]

    params['learning_rate']=0.05
    model1 = LGBMClassifier(**params)

    print('Processing Model1 ...')
    model1.fit(X_train,y_train,
               eval_set=[(X_valid,y_valid)],
               eval_metric='auc',
               verbose=False)

    params['learning_rate']=0.01
    model2 = LGBMClassifier(**params)

    print('Processing Model2 ...')
    model2.fit(X_train,y_train,
               eval_set=[(X_valid,y_valid)],
               eval_metric='auc',
               verbose=False,
               init_model=model1)

    params['learning_rate']=0.007
    model3 = LGBMClassifier(**params)

    print('Processing Model3 ...')
    model3.fit(X_train,y_train,
               eval_set=[(X_valid,y_valid)],
               eval_metric='auc',
               verbose=False,
               init_model=model2)

    params['learning_rate']=0.001
    model4 = LGBMClassifier(**params)

    print('Processing Model4 ...')
    model4.fit(X_train,y_train,
               eval_set=[(X_valid,y_valid)],
               eval_metric='auc',
               verbose=False,
               init_model=model3)

    pred_valid = model4.predict_proba(X_valid)[:,1]
    fpr, tpr, _ = roc_curve(y_valid, pred_valid)
    score = auc(fpr, tpr)
    scores.append(score)

    print(f"Fold: {fold + 1} Score: {score}")
    print('Predicting test data ...')

    test_preds.append(model3.predict_proba(X_test)[:,1])

print(f"Overall Validation Score: {np.mean(scores)}")


predictions = np.mean(np.column_stack(test_preds),axis=1)

sample_submission['target'] = predictions
sample_submission.to_csv('lgbm_sub_mean.csv', index=False)
sample_submission.head()


predictions = np.median(np.vstack(test_preds),axis=0)

sample_submission['target'] = predictions
sample_submission.to_csv('lgbm_sub_median.csv', index=False)
sample_submission.head()