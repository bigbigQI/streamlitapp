import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from mosaic.mosaic import Search
from env import Environment
from configuration_space import cs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import *
from imblearn.under_sampling import *
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import streamlit as st

dataname = st.sidebar.selectbox("选择数据集", ['camel-1.0.csv', 'alizadeh.csv', 'ant-1.7.csv', 'CM1.csv', 'colic.csv', 'ionosphere.csv'], key='1')
n_iteration = st.sidebar.slider("迭代次数", min_value=1, max_value=1000, value=100)
x = pd.read_csv("examples/"+dataname, encoding='utf-8', header=0)
show_data = st.sidebar.checkbox("展示数据集")
if show_data:
    st.write(x)

def svm_from_cfg(cfg, data):
    """ Creates a SVM based on a configuration and evaluates it on the
    iris-dataset using cross-validation.

    Parameters:
    -----------
    cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
        Configuration containing the parameters.
        Configurations are indexable!

    Returns:
    --------
    A crossvalidated mean score for the svm on the loaded data-set.
    """
    # For deactivated parameters, the configuration stores None-values.
    # This is not accepted by the SVM, so we remove them.
    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    # We translate boolean values:
    sampling_name = cfg["sampling"]
    cfg.pop('sampling', None)
    cfg["shrinking"] = True if cfg["shrinking"] == "true" else False
    # And for gamma, we set it to a fixed value or to "auto" (if used)
    if "gamma" in cfg:
        cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
        cfg.pop("gamma_value", None)  # Remove "gamma_value"
    x = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values
    

    sampling = SMOTE()
    if sampling_name == "ROS":
        sampling = RandomOverSampler()
    elif sampling_name == "SMOTE":
        sampling = SMOTE()
    elif sampling_name == "bSMOTE":
        sampling = BorderlineSMOTE()
    elif sampling_name == "ADASYN":
        sampling = ADASYN()
    elif sampling_name == "RUS":
        sampling = RandomUnderSampler()
    elif sampling_name == "Tomek-links":
        sampling = TomekLinks()
    elif sampling_name == "NearMiss":
        sampling = NearMiss()
    elif sampling_name == "CNN":
        sampling = CondensedNearestNeighbour()
    elif sampling_name == "OSS":
        sampling = OneSidedSelection()
    elif sampling_name == "NCR":
        sampling = NeighbourhoodCleaningRule()

    cv = StratifiedKFold(n_splits=5)
    results = []
    for train_idx, test_idx in cv.split(x, y):
        X_train, y_train = x[train_idx], y[train_idx]
        X_test, y_test = x[test_idx], y[test_idx]
        X_train, y_train = sampling.fit_resample(X_train, y_train)
        svm = SVC(**cfg, random_state=42, probability=True)
        svm.fit(X_train, y_train)
        y_pred_prob = svm.predict_proba(X_test)
        result = metrics.roc_auc_score(y_test, y_pred_prob[:,1])
        results.append(result)
    result = np.mean(results)
    print(result)           

    return result


def rf_from_cfg(cfg, data):
    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    # We translate boolean values:
    sampling_name = cfg["sampling"]
    cfg.pop('sampling', None)
    x = data
    x = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values

    sampling = SMOTE()
    if sampling_name == "ROS":
        sampling = RandomOverSampler()
    elif sampling_name == "SMOTE":
        sampling = SMOTE()
    elif sampling_name == "bSMOTE":
        sampling = BorderlineSMOTE()
    elif sampling_name == "ADASYN":
        sampling = ADASYN()
    elif sampling_name == "RUS":
        sampling = RandomUnderSampler()
    elif sampling_name == "Tomek-links":
        sampling = TomekLinks()
    elif sampling_name == "NearMiss":
        sampling = NearMiss()
    elif sampling_name == "CNN":
        sampling = CondensedNearestNeighbour()
    elif sampling_name == "OSS":
        sampling = OneSidedSelection()
    elif sampling_name == "NCR":
        sampling = NeighbourhoodCleaningRule()

    cv = StratifiedKFold(n_splits=5)
    results = []
    for train_idx, test_idx in cv.split(x, y):
        X_train, y_train = x[train_idx], y[train_idx]
        X_test, y_test = x[test_idx], y[test_idx]
        X_train, y_train = sampling.fit_resample(X_train, y_train)
        rf = RandomForestClassifier(**cfg, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_prob = rf.predict_proba(X_test)
        result = metrics.roc_auc_score(y_test, y_pred_prob[:,1])
        results.append(result)
    result = np.mean(results)
    print(result)           

    return result

pb = st.progress(0)
status_txt = st.empty()
chart = st.line_chart()
status_txt2 = st.empty()

environment = Environment(rf_from_cfg,
                          config_space=cs,
                          mem_in_mb=2048,
                          cpu_time_in_s=30,
                          seed=42,
                          data=x)

mosaic = Search(environment=environment,
                policy_arg = {"c_ucb": 1.1, "coef_progressive_widening": 0.6},
                verbose=True)
best_config, best_score = mosaic.run(nb_simulation=n_iteration,pb=pb, status_txt=status_txt, chart=chart, status_txt2=status_txt2)
print("Best config: ", best_config, "best score", best_score)
