#%%  Import necessary libraries for our analysis

#---Data processing and some useful libraries
#import pandasql as ps #such as sqldf in R
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


#---Visualisation
import matplotlib.pyplot as plt
import seaborn as sn 

#---Machine learning Algorithms and preparation of the dataset for ML algorithms
import sklearn
from sklearn.impute import  KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder,OrdinalEncoder,StandardScaler, Normalizer, normalize,MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict,cross_val_score, StratifiedKFold,GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from BorutaShap import BorutaShap, load_data
from sklearn import metrics  #Additional scklearn functions
from xgboost import cv
from xgboost.sklearn import XGBClassifier
from sklearn.pipeline import make_pipeline
import optuna## Optuna Hyperparameter Search method
# Skopt functions
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Categorical, Integer
#-- -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
