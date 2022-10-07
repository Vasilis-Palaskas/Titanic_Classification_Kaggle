
""" Hyperparameter tuning using nested cross validation scheme through Bayes Search method
    (both hyperparameters and data processing pipelines)  using Random Forest Classifier.

"""

#  Instance of a XGB Classifier object
rnf_initial=RandomForestClassifier(n_jobs=-1,
                          random_state=52)
# Combine into a single one
# Data Processing Pipelines 1,2 : Combine data processing pipelines (categorical+ordinal+numeric) 
#                                       where the numeric features are standardized, normalise, respectively.                                
pipeline_1_2 = ColumnTransformer([("cat_pipeline",cat_pipeline_adv_val, cat_vars),                                   
                                 ("num_pipeline",num_pipeline_1_2, num_vars)])# std scaler or normaliser for numeric

# Sequentially, combine  pipelines by combining either data processing pipelines 1 or 2 with the RF Classifier estimator.
rnf_clf_pipeline_1_2 = Pipeline(steps=[
    ('col_trans', pipeline_1_2),
    ('model',rnf_initial) ])

# --- Define nested cross-validation scheme
cv_outer = StratifiedKFold(n_splits=10,shuffle=True, random_state=52)
cv_inner = StratifiedKFold(n_splits=6,shuffle=True, random_state=52) # Future consideration: RepeatedStratifiedKFold(n_repeats=3,n_splits=10,random_state=0)
                        
# -- RF Classifier (Random Forest) hyperparams distributions to search
rnf_params= {
    # Number of boosted trees to fit
    'model__n_estimators':Integer(50,1990),
    "model__min_samples_split": Integer( 2,40),
    "model__min_samples_leaf": Integer( 1,30),
    "model__max_depth": Integer(2, 92),
    #"model__subsample": Real(0.3,1,'uniform'),
    "model__max_features": Categorical( ["auto", "sqrt"]),
    "model__bootstrap" : Categorical( ["True", "False"]),
    #"model__colsample_bytree": Real(0.3,1,'uniform'),
    # L2 regularization
    #'model__reg_lambda':Real(1e-2,100,'log-uniform'),
    # L1 regularisation
    #'model__reg_alpha':Real(1e-2,100,'log-uniform')
        }

# Bayes Search Classifier within nested-cross validation scheme (see next block of code) 
# will search combinations of the classifier hyperparameters 
# along with the data processing pipelines 1,2 to find the best results.
grid_rnf_params = [{**{'col_trans__num_pipeline__normaliser': ['passthrough']}, **rnf_params},
                    {**{'col_trans__num_pipeline__scaler': ['passthrough']}, **rnf_params}]

#%% Nested cross validation block (xgb acronym will be used for any result-process related to the XGB Classifier)

# out and in sample accuracies for cv outer splits
outer_results = list()
in_sample_acc_results = list()

# out and in sample f1 scores for cv outer splits
outer_results_f1 = list()
in_sample_f1_results = list()

# out and in sample predictions for cv outer splits
outer_yhat_bayessearach_rnf =list()
outer_yhat_train_bayessearach_rnf =list()

# best params and score for cv inner splits
inner_best_params=list()
inner_best_score=list()

#---Competittion test dataset predictions
y_out_sample_test_results=list()

# Now the nested cross validation begins
i=0
for train_ix, test_ix in cv_outer.split(X,y):
    i=i+1
    print(i)
    # split data
    X_train = X.iloc[train_ix,:]
    X_test = X.iloc[test_ix,:]
    y_train = y.iloc[train_ix]
    y_test = y.iloc[test_ix]

    #-------- Hyperparameters search through the Bayes Search Method in inner splits (XGBClassifier)
    np.random.seed(1234)
    # object related to the Bayes Search implemented in each cv-inner split
    opt_inner =BayesSearchCV(estimator=rnf_clf_pipeline_1_2,search_spaces=grid_rnf_params,
                      scoring='accuracy',cv=cv_inner,refit=True,
                      return_train_score=False, 
                      # Gaussian Processes (GP) As surrogate/proxy function
                      optimizer_kwargs={'base_estimator':'GP'},n_iter=35,
                      random_state=52)
    # Ignore warnings
    warnings.filterwarnings("ignore", category=DataConversionWarning)#
    warnings.filterwarnings("ignore", category=DeprecationWarning)#eliminate warnings for deprecation reasons
	
    # Fit the object related to the Bayes Search cross validation in cv inner splits
    grid_hyperparams_inner_result=opt_inner.fit(X_train, y_train)

    # get the best performing model fit by the Bayes Search method implemented in inner-k-fold splits.
    best_model = grid_hyperparams_inner_result.best_estimator_
    best_parameters =grid_hyperparams_inner_result.best_params_
    best_score =grid_hyperparams_inner_result.best_score_
    
    # evaluate model on the hold out dataset of each cv outer split
    yhat = best_model.predict(X_test)
    acc = accuracy_score(y_test, yhat).round(2)*100 
    f1_out=f1_score(y_test, yhat).round(2)*100  # 
   
    # evaluate model on the hold in dataset of each cv outer split
    yhat_train = best_model.predict(X_train)
    in_sample_acc=accuracy_score(y_train, yhat_train).round(2)*100 
    f1_in=f1_score(y_train, yhat_train).round(2)*100  # 

    # store the cv outer results
    outer_results.append(acc)
    outer_results_f1.append(f1_out)
    outer_yhat_bayessearach_rnf.append(yhat)
    outer_yhat_train_bayessearach_rnf.append(yhat_train)
    # store the cv inner results
    in_sample_acc_results.append(in_sample_acc)
    in_sample_f1_results.append(f1_in)
    inner_best_params.append(best_parameters)
    inner_best_score.append(grid_hyperparams_inner_result.best_score_)
    # Public Leaderboard test dataset predictions using nested-cross validation models' predictions
    y_out_sample_test = best_model.predict(new_titanic_test_data)
    y_out_sample_test_results.append(y_out_sample_test)
    # report progress per cv outer split with the best parameters in each cv outer split
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc,best_score, 
                                           best_parameters))

# summarize the estimated performance of the model
print('Out-Sample Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results))) #Accuracy:82.300 (3.951) 
print('Out-Sample F1 score: %.3f (%.3f)' % (np.mean(outer_results_f1), np.std(outer_results_f1))) #F1: 75.600 (5.004)

print('In-Sample Accuracy: %.3f (%.3f)' % (np.mean(in_sample_acc_results), np.std(in_sample_acc_results))) #Accuracy: 88.200 (1.600)
print('In-Sample F1 score: %.3f (%.3f)' % (np.mean(in_sample_f1_results), np.std(in_sample_f1_results))) #F1:83.600 (2.245)

#%%%% Save model training and model prediction results both inner and outer cv splits 

#---Choose working directory

directory='C:/Users/vasileios palaskas/Documents/GitHub/Titanic_Classification_Kaggle/models/RF Predictions'
logger.info('Define the directory of your saved model objects (folder Titanic_Classification_Kaggle/models/')
os.chdir(directory)


# Write-Save objects related to the Nested Cross-validation prodedure implemented through Random Forest Classifier

# CV outer splits: Out-sample results (accuracies)
with open("outer_results_bayessearach_rnf", "wb") as outer_results_fp:   #Pickling
 pickle.dump(outer_results, outer_results_fp)
 
 # CV outer splits: In-sample results (accuracies)
with open("in_sample_acc_results_bayessearach_rnf", "wb") as in_sample_acc_results_fp:   #Pickling
  pickle.dump(in_sample_acc_results, in_sample_acc_results_fp)

# CV outer splits: Out-sample predictions of the response 
with open("outer_yhat_bayessearach_rnf", "wb") as outer_yhat_bayessearach_rnf_fp:   #Pickling
  pickle.dump(outer_yhat_bayessearach_rnf, outer_yhat_bayessearach_rnf_fp)
  
# CV outer splits: Public Leaderboard test dataset predictions of the response   
with open("y_out_sample_test_results_bayessearach_rnf", "wb") as y_out_sample_test_results_fp:   #Pickling
     pickle.dump(y_out_sample_test_results, y_out_sample_test_results_fp)



# Load objects

# CV outer splits: Out-sample results (accuracies)
 with open("in_sample_acc_results_bayessearach_rnf", "rb") as in_sample_acc_results_fp:   # Unpickling
  in_sample_acc_results_bayessearach_rnf = pickle.load(in_sample_acc_results_fp)

# CV outer splits: In-sample results (accuracies)
with open("outer_yhat_bayessearach_rnf", "rb") as outer_yhat_bayessearach_rnf_fp:   # Unpickling
        yhat_bayessearach_rnf_list = pickle.load(outer_yhat_bayessearach_rnf_fp)

# CV outer splits: Out-sample predictions of the response      
with open("outer_results_bayessearach_rnf", "rb") as outer_results_fp:   # Unpickling
 outer_results_bayessearach_rnf = pickle.load(outer_results_fp)

# CV outer splits: Public Leaderboard test dataset predictions of the response         
with open("y_out_sample_test_results_bayessearach_rnf", "rb") as y_out_sample_test_results_fp:   # Unpickling
 y_out_sample_test_results_bayessearach_rnf_list = pickle.load(y_out_sample_test_results_fp) 

         
 
