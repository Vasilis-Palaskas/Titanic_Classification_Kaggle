
#%% Preparation for the hyperparams search and cross validation scheme

# ---- Data processing pipelines& Classifier combination

# Define classifiers
rnf_initial=RandomForestClassifier(n_jobs=-1,
                          random_state=52)

# Data processing pipelines to be compared during nested cross-val.
num_pipeline_1_2=Pipeline( [('scaler',  StandardScaler()),
                            ('normaliser', MinMaxScaler()),
                            ('imputer', KNNImputer(n_neighbors=10))] )
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent') ),
                            ('encoder', OneHotEncoder(handle_unknown='ignore',drop = 'first'))])

ord_pipeline=Pipeline([('imputer', SimpleImputer(strategy='most_frequent') ),
                            ('encoder', OrdinalEncoder())])
pipeline_1_2 = ColumnTransformer([("cat_pipeline", cat_pipeline, cat_vars),                                   
                                 ("num_pipeline",num_pipeline_1_2, num_vars)])# std scaler or normaliser for numeric
# Combin pipeline+classifier
rnf_clf_pipeline_1_2 = Pipeline(steps=[
    ('col_trans', pipeline_1_2),
    ('model',rnf_initial) ])

# --- Define nested cross-validation scheme
cv_outer = StratifiedKFold(n_splits=10,shuffle=True, random_state=52)
cv_inner = StratifiedKFold(n_splits=6,shuffle=True, random_state=52) # Future consideration: RepeatedStratifiedKFold(n_repeats=3,n_splits=10,random_state=0)
                        
# Define parameter space for search depending on the specific classifier

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


grid_rnf_params = [{**{'col_trans__num_pipeline__normaliser': ['passthrough']}, **rnf_params},
                    {**{'col_trans__num_pipeline__scaler': ['passthrough']}, **rnf_params}]


#--- Enumerate splits and initialise a vector with results
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
i=0
# Now the nested cross validation begins
for train_ix, test_ix in cv_outer.split(X,y):
    i=i+1
    print(i)
    # split data
    X_train = X.iloc[train_ix,:]
    X_test = X.iloc[test_ix,:]
    y_train = y.iloc[train_ix]
    y_test = y.iloc[test_ix]

    #-------- Hyperparameters search in inner splits (XGBClassifier)
	# create grid search
    np.random.seed(1234)

    opt_inner =BayesSearchCV(estimator=rnf_clf_pipeline_1_2,search_spaces=grid_rnf_params,
                      scoring='accuracy',cv=cv_inner,refit=True,
                      return_train_score=False, 
                      # Gaussian Processes (GP) As surrogate/proxy function
                      optimizer_kwargs={'base_estimator':'GP'},n_iter=35,
                      random_state=52)
    # Ignore warnings
    warnings.filterwarnings("ignore", category=DataConversionWarning)#
    warnings.filterwarnings("ignore", category=DeprecationWarning)#eliminate warnings for deprecation reasons
	# execute search
    grid_hyperparams_inner_result=opt_inner.fit(X_train, y_train)
    #print(pd.DataFrame(grid_hyperparams_inner_result.cv_results_))

    # get the best performing model fit by grid-searching in inner-k-fold splits of the train dataset from outer split.
    best_model = grid_hyperparams_inner_result.best_estimator_
    best_parameters =grid_hyperparams_inner_result.best_params_
    best_score =grid_hyperparams_inner_result.best_score_
    # evaluate model on the hold out dataset
    yhat = best_model.predict(X_test)
    acc = accuracy_score(y_test, yhat).round(2)*100 
    f1_out=f1_score(y_test, yhat).round(2)*100  # 
    # evaluate model on the hold in dataset
    yhat_train = best_model.predict(X_train)
    in_sample_acc=accuracy_score(y_train, yhat_train).round(2)*100 
    f1_in=f1_score(y_train, yhat_train).round(2)*100  # 

    # store the result
    outer_results.append(acc)
    outer_results_f1.append(f1_out)


    outer_yhat_bayessearach_rnf.append(yhat)
    outer_yhat_train_bayessearach_rnf.append(yhat_train)
    
    in_sample_acc_results.append(in_sample_acc)
    in_sample_f1_results.append(f1_in)
    # store the inner results

    inner_best_params.append(best_parameters)
    inner_best_score.append(grid_hyperparams_inner_result.best_score_)
    # test predictions for the competitions using nested-cross val runs
    y_out_sample_test = best_model.predict(new_titanic_test_data)
    y_out_sample_test_results.append(y_out_sample_test)
    # report progress
    print('>acc=%.3f, est=%.3f, cfg=%s' % (acc,best_score, 
                                           best_parameters))

# summarize the estimated performance of the model
print('Out-Sample Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results))) #Accuracy:82.300 (3.951) 
print('Out-Sample F1 score: %.3f (%.3f)' % (np.mean(outer_results_f1), np.std(outer_results_f1))) #F1: 75.600 (5.004)

print('In-Sample Accuracy: %.3f (%.3f)' % (np.mean(in_sample_acc_results), np.std(in_sample_acc_results))) #Accuracy: 88.200 (1.600)
print('In-Sample F1 score: %.3f (%.3f)' % (np.mean(in_sample_f1_results), np.std(in_sample_f1_results))) #F1:83.600 (2.245)

#%%%% Save model training and model prediction results both inner and outer cv splits 
with open("outer_results_bayessearach_rnf", "wb") as outer_results_fp:   #Pickling
 pickle.dump(outer_results, outer_results_fp)

with open("outer_results_bayessearach_rnf", "rb") as outer_results_fp:   # Unpickling
 outer_results_bayessearach_rnf = pickle.load(outer_results_fp)
#
 with open("in_sample_acc_results_bayessearach_rnf", "wb") as in_sample_acc_results_fp:   #Pickling
  pickle.dump(in_sample_acc_results, in_sample_acc_results_fp)

 with open("in_sample_acc_results_bayessearach_rnf", "rb") as in_sample_acc_results_fp:   # Unpickling
  in_sample_acc_results_bayessearach_rnf = pickle.load(in_sample_acc_results_fp)
 #
 with open("outer_yhat_bayessearach_rnf", "wb") as outer_yhat_bayessearach_rnf_fp:   #Pickling
  pickle.dump(outer_yhat_bayessearach_rnf, outer_yhat_bayessearach_rnf_fp)
with open("outer_yhat_bayessearach_rnf", "rb") as outer_yhat_bayessearach_rnf_fp:   # Unpickling
        yhat_bayessearach_rnf_list = pickle.load(outer_yhat_bayessearach_rnf_fp)
#%%%% Store our predictions (saved in models/RF_Predictions)

with open("y_out_sample_test_results_bayessearach_rnf", "wb") as y_out_sample_test_results_fp:   #Pickling
 pickle.dump(y_out_sample_test_results, y_out_sample_test_results_fp)

with open("y_out_sample_test_results_bayessearach_rnf", "rb") as y_out_sample_test_results_fp:   # Unpickling
 y_out_sample_test_results_bayessearach_rnf_list = pickle.load(y_out_sample_test_results_fp) 

         
 
