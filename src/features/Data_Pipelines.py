""" In this script, we prepare some data processing pipelines in order to avoid calling it
 repeatedly in each script.
"""
#%% Data Processing Pipelines for Feature Importance Analysis (implemented in this directory /src/Feature_Importance.py)

""" 1): Data Processing Pipeline required for the Feature Importance Algorithm
"""
#---Categorical Variables
cat_pipeline_feat_import = Pipeline([('imputer', SimpleImputer(strategy='most_frequent') ),
                            ('encoder', OneHotEncoder(handle_unknown='ignore',drop = 'first'))])
#---Ordinal Variables
ord_pipeline_feat_import=Pipeline([('imputer', SimpleImputer(strategy='most_frequent') ),
                            ('encoder', OrdinalEncoder())])


#--- Numeric Pipeline 
num_pipeline_feat_import= Pipeline( [('scaler',  StandardScaler())] )


#%%   Data Processing Pipelines required for the adversarial validation techniques (.. /src/Adversarial_validation/adversarial_valid.py)

""" 2): Data Processing Pipeline required for  the adversarial validation techniques
"""
#---Categorical Variables

cat_pipeline_adv_val = Pipeline([('imputer', SimpleImputer(strategy='most_frequent') ),
                        ('encoder', OneHotEncoder(handle_unknown='ignore',drop = 'first'))])
#---Ordinal Variables

ord_pipeline_adv_val=Pipeline([('imputer', SimpleImputer(strategy='most_frequent') ),
                        ('encoder', OrdinalEncoder())])
#--- Numeric Pipelines 

num_pipeline1= Pipeline( [('scaler',  StandardScaler()),
                        ('imputer', KNNImputer(n_neighbors=10))] )# standard scaler

num_pipeline2= Pipeline( [('normalizer',  MinMaxScaler()),
                            ('imputer', KNNImputer(n_neighbors=10))] )# normaliser scaler



#%%   Data Processing Pipelines required for the hyperparameter tuning process
#--   within nested cross-validation framework across several fitted Classifiers (XGBoost, Random Forest, ANN through Keras Classifier from sklearn API)
""" 3): Data Processing Pipeline required for the the hyperparameter tuning process 
             across several fitted Classifiers.
"""

# Two Data processing sequential pipelines. Those can be compared during nested cross-val.
# More specifically, we will test either Standard Scaller or Normalizer as data processing pipeline
# along with Imputer
num_pipeline_1_2=Pipeline( [('scaler',  StandardScaler()),
                            ('normaliser', MinMaxScaler()),
                            ('imputer', KNNImputer(n_neighbors=10))] )

