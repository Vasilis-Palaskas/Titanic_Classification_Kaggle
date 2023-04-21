#%%Libraries Loading
from sklearn.linear_model import LogisticRegression
#%% Preparation for the hyperparams search and cross validation scheme

# ---- Data processing pipelines& Classifier combination

# Define classifiers
logisticclas_initial=LogisticRegression( random_state=52)

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
logisticclas_clf_pipeline_1_2 = Pipeline(steps=[
    ('col_trans', pipeline_1_2),
    ('model',logisticclas_initial) ])

# Define parameter space for search depending on the specific classifier

logisticclas_params= {
    # Number of boosted trees to fit
    'model__solver':Categorical( ['newton-cg', 'lbfgs', 'liblinear']),
    "model__penalty": Categorical( ['l2']),
    "model__C": Real(1e-2,100,'log-uniform')
        }

#np.logspace(-3,3,7)
grid_logisticclas_params = [{**{'col_trans__num_pipeline__normaliser': ['passthrough']}, **logisticclas_params},
                    {**{'col_trans__num_pipeline__scaler': ['passthrough']}, **logisticclas_params}]


#---Bayesearch with loop for XGB Classifier
# Enumerate splits and initialise a vector with results
outer_results = list()
in_sample_acc_results = list()
outer_results_f1 = list()
in_sample_f1_results = list()
inner_best_params=list()
inner_best_score=list()
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

    opt_inner =BayesSearchCV(estimator=logisticclas_clf_pipeline_1_2,search_spaces=grid_logisticclas_params,
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
print('Out-Sample Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results))) #Accuracy:80.500 (3.138)
print('Out-Sample F1 score: %.3f (%.3f)' % (np.mean(outer_results_f1), np.std(outer_results_f1))) #F1: 70.000 (4.382)

print('In-Sample Accuracy: %.3f (%.3f)' % (np.mean(in_sample_acc_results), np.std(in_sample_acc_results))) #Accuracy:80.900 (0.700)
print('In-Sample F1 score: %.3f (%.3f)' % (np.mean(in_sample_f1_results), np.std(in_sample_f1_results))) #F1: 70.500 (0.922)
inner_best_score
inner_best_params

outer_results
outer_results_f1

in_sample_acc
in_sample_f1_results
