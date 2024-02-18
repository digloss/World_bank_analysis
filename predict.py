import pandas as pd
import numpy as np

import pycountry, pycountry_convert

import seaborn as sns




from sklearn.linear_model import ElasticNet

# KNN regression
from sklearn.neighbors import KNeighborsRegressor, KernelDensity, KDTree

# XGBoost
import xgboost as xgb

# Neural Networks regression
from tensorflow.keras import Model, Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from tensorflow.keras.layers import Dense, Dropout, LSTM, LeakyReLU, BatchNormalization
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
import tensorflow as tf
from tensorflow.keras.utils import plot_model
tf.compat.v1.disable_eager_execution()
from bayes_opt import BayesianOptimization
import warnings
warnings.filterwarnings('ignore')

# Walking window CV:
from sklearn.model_selection._split import TimeSeriesSplit


from math import sqrt, log, exp, floor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedKFold, RandomizedSearchCV, GridSearchCV, cross_validate, cross_val_predict
from sklearn.impute import KNNImputer
from sklearn import metrics
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error # for error reporting
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import matplotlib
import statsmodels.api as sm
# matplotlib.style.use('seaborn-v0_8')