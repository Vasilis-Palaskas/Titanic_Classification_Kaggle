#%%  Import necessary libraries for this project. 

# Below, we import libraries depending on each functionality/task we will implement in the next Python scripts

#---Data processing and some useful libraries
import requests
import os
import warnings
import pickle# save objects
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)#e
warnings.filterwarnings("ignore", category=DeprecationWarning)#eliminate warnings for deprecation reasons
import numpy as np 
import pandas as pd
import re
from time import time
import pprint
import joblib
from functools import partial
import tkinter as tk
from  tkinter  import filedialog# library for choosing manually working directory
from sklearn.impute import  KNNImputer, SimpleImputer
#---Visualisation
import matplotlib.pyplot as plt
import seaborn as sn 

#---Machine-Deep learning Algorithms 
import sklearn
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder,OrdinalEncoder,StandardScaler, Normalizer, normalize,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier

import tensorflow as tf #pip install tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from scikeras.wrappers import KerasClassifier

#--- Cross Validation-Model Selection scores-Pipelines Implementation-Hyperparameters Search
from sklearn.model_selection import cross_val_predict,cross_val_score, StratifiedKFold,GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from BorutaShap import BorutaShap, load_data
from sklearn import metrics  #Additional scklearn functions
from xgboost import cv

from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

import optuna## Optuna  Search method
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Categorical, Integer


#-- -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
