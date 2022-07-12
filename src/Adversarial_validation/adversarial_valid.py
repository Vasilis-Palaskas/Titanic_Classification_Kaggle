

#-----------Adversarial validation to check concept/distribution shift issues-------------
""" Runs adversarial validation technique to check whether the final data structure
        encounters concept/distribution shift issues
"""

#%% Features Selections
#--Case 1: Use features emerged from the Boruta (Feature Selection) process in Feature_Importance.py

# Grouping of variables
cat_vars=["title","Sex"]
ord_vars=[]
num_vars=["Age","Pclass"]
all_variables=["title","Sex","Age","Pclass"]

new_titanic_train_data=pd.DataFrame(titanic_train_data[all_variables])
new_titanic_test_data=pd.DataFrame(titanic_train_data[all_variables])

new_titanic_train_data.head(10)
new_titanic_test_data.head(10)

X=pd.DataFrame(new_titanic_train_data[all_variables])
y=pd.DataFrame(titanic_train_data['Survived'].values)
#%%% Data Processing and Classifier Pipelines Formation

#-Initial Random Forest Classifier
model=RandomForestClassifier(random_state = 1)

#--Pipeline 1
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent') ),
                        ('encoder', OneHotEncoder(handle_unknown='ignore',drop = 'first'))])

ord_pipeline=Pipeline([('imputer', SimpleImputer(strategy='most_frequent') ),
                        ('encoder', OrdinalEncoder())])

num_pipeline1= Pipeline( [('scaler',  StandardScaler()),
                        ('imputer', KNNImputer(n_neighbors=10))] )

# combine pipelines
pipeline1 = ColumnTransformer([("cat_pipeline", cat_pipeline, cat_vars),
                                        ("ord_pipeline", ord_pipeline, ord_vars),
                                       ("num_pipeline",num_pipeline1, num_vars)])
clf_pipeline1 = Pipeline(steps=[
    ('col_trans', pipeline1),
    ('model',model)])

#--Pipeline 2

num_pipeline2= Pipeline( [('normalizer',  MinMaxScaler()),
                            ('imputer', KNNImputer(n_neighbors=10))] )
# combine pipelines
pipeline2 = ColumnTransformer([("cat_pipeline", cat_pipeline, cat_vars),
                                        ("ord_pipeline", ord_pipeline, ord_vars),
                                       ("num_pipeline",num_pipeline2, num_vars)])

clf_pipeline2 = Pipeline(steps=[
    ('col_trans', pipeline2),
    ('model',model)])

#-Define nested cross-validation scheme
cv_outer = StratifiedKFold(n_splits=10,shuffle=True, random_state=52)


#%%%% Adversarial validation using nested cross-validation scheme (our strategy will use nested cross-val and for this reason we check it here)
logger.info('Adversarial validation begins using the same cross-validation scheme that we will use for hyperparams search and model training')
#--- Enumerate splits and initialise a vector with results for ROC-AUC pre cv outer split
roc_auc_cv_outer_results= list()#roc-auc vector between train and test sets of cv outer splits
roc_auc_cv_results_public_leaderboard= list()#roc-auc vector between train sets of cv outer splits and test dataset of public leaderboard

i=0# iterator
for train_ix, test_ix in cv_outer.split(X,y):# Now the nested cross validation begins
    print(i)
    # split data
    X_train = X.iloc[train_ix,:]
    X_test = X.iloc[test_ix,:]
    y_train = y.iloc[train_ix]
    y_test = y.iloc[test_ix]
    #----adversarial validation between x_train and x_test from cv outer splits
    train_adv_val=X_train
    test_adv_val=X_test
    #  y here is the index of belonging or not to the train set 
    X_adv=train_adv_val.append(test_adv_val)
    y_adv=[0]*len(train_adv_val)+[1]*len(test_adv_val)
    np.random.seed(1234)
    # cross validation
    cv_preds=cross_val_predict(clf_pipeline1,X_adv,y_adv,cv=5,n_jobs=-1,method="predict_proba")
    print(roc_auc_score(y_true=y_adv,y_score=cv_preds[:,1]))# 
    roc_auc_cv_outer_results.append(roc_auc_score(y_true=y_adv,y_score=cv_preds[:,1]))
    #--adversarial validation between x_train from cv outer splits and out_sample test set of the public leaderboard
    # create train and test sets
    train_adv_val_public_leaderboard=X_train
    test_adv_val_public_leaderboard=new_titanic_test_data
    # Union of X and y where y here is the index of belonging or not to the train set 
    X_adv_public_leaderboard=train_adv_val_public_leaderboard.append(test_adv_val_public_leaderboard)
    y_adv_public_leaderboard=[0]*len(train_adv_val_public_leaderboard)+[1]*len(test_adv_val_public_leaderboard)
    np.random.seed(1234)
    # cross validation

    cv_preds_public_leaderboard=cross_val_predict(clf_pipeline1,X_adv_public_leaderboard,
                                                   y_adv_public_leaderboard,cv=5,n_jobs=-1,method="predict_proba")
    #model_fit=model.fit(X,y)
    print(roc_auc_score(y_true=y_adv_public_leaderboard,
                        y_score=cv_preds_public_leaderboard[:,1]))#  0.67 ROC -AUC score for our predictions
    roc_auc_cv_results_public_leaderboard.append(roc_auc_score(y_true=y_adv_public_leaderboard,
                                                                y_score=cv_preds_public_leaderboard[:,1]))
    i=i+1

print('Pipeline 1: Roc-Auc average score between train and test sets of cv outer splits: %.3f (%.3f)' % (np.mean(roc_auc_cv_outer_results), 
                                         np.std(roc_auc_cv_outer_results))) #0.500 (0.042)
print('Pipeline 1: Roc-Auc average score between train from cv outer splits and public learderboard test sets: %.3f (%.3f)' % (np.mean(roc_auc_cv_results_public_leaderboard),
                                                  np.std(roc_auc_cv_results_public_leaderboard))) #0.504 (0.008)

#--- Enumerate splits and initialise a vector with results for ROC-AUC pre cv outer split
roc_auc_cv_outer_results= list()#roc-auc vector between train and test sets of cv outer splits
roc_auc_cv_results_public_leaderboard= list()#roc-auc vector between train sets of cv outer splits and test dataset of public leaderboard

i=0# iterator
for train_ix, test_ix in cv_outer.split(X,y):# Now the nested cross validation begins
    print(i)
    # split data
    X_train = X.iloc[train_ix,:]
    X_test = X.iloc[test_ix,:]
    y_train = y.iloc[train_ix]
    y_test = y.iloc[test_ix]
    #----adversarial validation between x_train and x_test from cv outer splits
    train_adv_val=X_train
    test_adv_val=X_test
    #  y here is the index of belonging or not to the train set 
    X_adv=train_adv_val.append(test_adv_val)
    y_adv=[0]*len(train_adv_val)+[1]*len(test_adv_val)
    np.random.seed(1234)
    # cross validation
    cv_preds=cross_val_predict(clf_pipeline2,X_adv,y_adv,cv=5,n_jobs=-1,method="predict_proba")
    print(roc_auc_score(y_true=y_adv,y_score=cv_preds[:,1]))# 
    roc_auc_cv_outer_results.append(roc_auc_score(y_true=y_adv,y_score=cv_preds[:,1]))
    #--adversarial validation between x_train from cv outer splits and out_sample test set of the public leaderboard
    # create train and test sets
    train_adv_val_public_leaderboard=X_train
    test_adv_val_public_leaderboard=new_titanic_test_data
    # Union of X and y where y here is the index of belonging or not to the train set 
    X_adv_public_leaderboard=train_adv_val_public_leaderboard.append(test_adv_val_public_leaderboard)
    y_adv_public_leaderboard=[0]*len(train_adv_val_public_leaderboard)+[1]*len(test_adv_val_public_leaderboard)
    np.random.seed(1234)
    # cross validation

    cv_preds_public_leaderboard=cross_val_predict(clf_pipeline2,X_adv_public_leaderboard,
                                                   y_adv_public_leaderboard,cv=5,n_jobs=-1,method="predict_proba")
    #model_fit=model.fit(X,y)
    print(roc_auc_score(y_true=y_adv_public_leaderboard,
                        y_score=cv_preds_public_leaderboard[:,1]))#  0.67 ROC -AUC score for our predictions
    roc_auc_cv_results_public_leaderboard.append(roc_auc_score(y_true=y_adv_public_leaderboard,
                                                                y_score=cv_preds_public_leaderboard[:,1]))
    i=i+1

print('Pipeline 2: Roc-Auc average score between train and test sets of cv outer splits: %.3f (%.3f)' % (np.mean(roc_auc_cv_outer_results), 
                                         np.std(roc_auc_cv_outer_results))) #0.505 (0.036)
print('Pipeline 2: Roc-Auc average score between train from cv outer splits and public learderboard test sets: %.3f (%.3f)' % (np.mean(roc_auc_cv_results_public_leaderboard),
                                                  np.std(roc_auc_cv_results_public_leaderboard))) #0.489 (0.004)


#%% Features Selections


#--Case 2: Use features emerged from EDA process in visualisation_descriptive_analysis.py

# Grouping of variables
# Grouping of variables
cat_vars=["Embarked","Sex","is_alone","title","Age_new"]
ord_vars=[]
num_vars=["sqrt_fare","Pclass"]
all_variables=["Embarked","Sex","is_alone","title","Age_new","sqrt_fare","Pclass"]

new_titanic_train_data=pd.DataFrame(titanic_train_data[all_variables])
new_titanic_test_data=pd.DataFrame(titanic_train_data[all_variables])

new_titanic_train_data.head(10)
new_titanic_test_data.head(10)
X=pd.DataFrame(new_titanic_train_data)
y=pd.DataFrame(titanic_train_data['Survived'].values)
#-Initial Random Forest Classifier
model=RandomForestClassifier(random_state = 1)

#--Pipeline 1
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent') ),
                        ('encoder', OneHotEncoder(handle_unknown='ignore',drop = 'first'))])

ord_pipeline=Pipeline([('imputer', SimpleImputer(strategy='most_frequent') ),
                        ('encoder', OrdinalEncoder())])

num_pipeline1= Pipeline( [('scaler',  StandardScaler()),
                        ('imputer', KNNImputer(n_neighbors=10))] )

# combine pipelines
pipeline1 = ColumnTransformer([("cat_pipeline", cat_pipeline, cat_vars),
                                        ("ord_pipeline", ord_pipeline, ord_vars),
                                       ("num_pipeline",num_pipeline1, num_vars)])
clf_pipeline1 = Pipeline(steps=[
    ('col_trans', pipeline1),
    ('model',model)])

#--Pipeline 2

num_pipeline2= Pipeline( [('normalizer',  MinMaxScaler()),
                            ('imputer', KNNImputer(n_neighbors=10))] )
# combine pipelines
pipeline2 = ColumnTransformer([("cat_pipeline", cat_pipeline, cat_vars),
                                        ("ord_pipeline", ord_pipeline, ord_vars),
                                       ("num_pipeline",num_pipeline2, num_vars)])

clf_pipeline2 = Pipeline(steps=[
    ('col_trans', pipeline2),
    ('model',model)])


#%%%% Adversarial validation using nested cross-validation scheme (our strategy will use nested cross-val and for this reason we check it here)

#--- Enumerate splits and initialise a vector with results for ROC-AUC pre cv outer split
roc_auc_cv_outer_results= list()#roc-auc vector between train and test sets of cv outer splits
roc_auc_cv_results_public_leaderboard= list()#roc-auc vector between train sets of cv outer splits and test dataset of public leaderboard

i=0# iterator
for train_ix, test_ix in cv_outer.split(X,y):# Now the nested cross validation begins
    print(i)
    # split data
    X_train = X.iloc[train_ix,:]
    X_test = X.iloc[test_ix,:]
    y_train = y.iloc[train_ix]
    y_test = y.iloc[test_ix]
    #----adversarial validation between x_train and x_test from cv outer splits
    train_adv_val=X_train
    test_adv_val=X_test
    #  y here is the index of belonging or not to the train set 
    X_adv=train_adv_val.append(test_adv_val)
    y_adv=[0]*len(train_adv_val)+[1]*len(test_adv_val)
    np.random.seed(1234)
    # cross validation
    cv_preds=cross_val_predict(clf_pipeline1,X_adv,y_adv,cv=5,n_jobs=-1,method="predict_proba")
    print(roc_auc_score(y_true=y_adv,y_score=cv_preds[:,1]))# 
    roc_auc_cv_outer_results.append(roc_auc_score(y_true=y_adv,y_score=cv_preds[:,1]))
    #--adversarial validation between x_train from cv outer splits and out_sample test set of the public leaderboard
    # create train and test sets
    train_adv_val_public_leaderboard=X_train
    test_adv_val_public_leaderboard=new_titanic_test_data
    # Union of X and y where y here is the index of belonging or not to the train set 
    X_adv_public_leaderboard=train_adv_val_public_leaderboard.append(test_adv_val_public_leaderboard)
    y_adv_public_leaderboard=[0]*len(train_adv_val_public_leaderboard)+[1]*len(test_adv_val_public_leaderboard)
    np.random.seed(1234)
    # cross validation

    cv_preds_public_leaderboard=cross_val_predict(clf_pipeline1,X_adv_public_leaderboard,
                                                   y_adv_public_leaderboard,cv=5,n_jobs=-1,method="predict_proba")
    #model_fit=model.fit(X,y)
    print(roc_auc_score(y_true=y_adv_public_leaderboard,
                        y_score=cv_preds_public_leaderboard[:,1]))#  0.67 ROC -AUC score for our predictions
    roc_auc_cv_results_public_leaderboard.append(roc_auc_score(y_true=y_adv_public_leaderboard,
                                                                y_score=cv_preds_public_leaderboard[:,1]))
    i=i+1

print('Pipeline 1: Roc-Auc average score between train and test sets of cv outer splits: %.3f (%.3f)' % (np.mean(roc_auc_cv_outer_results), 
                                         np.std(roc_auc_cv_outer_results))) #0.535 (0.034)
print('Pipeline 1: Roc-Auc average score between train from cv outer splits and public learderboard test sets: %.3f (%.3f)' % (np.mean(roc_auc_cv_results_public_leaderboard),
                                                  np.std(roc_auc_cv_results_public_leaderboard))) #0.488 (0.006)

#--- Enumerate splits and initialise a vector with results for ROC-AUC pre cv outer split
roc_auc_cv_outer_results= list()#roc-auc vector between train and test sets of cv outer splits
roc_auc_cv_results_public_leaderboard= list()#roc-auc vector between train sets of cv outer splits and test dataset of public leaderboard

i=0# iterator
for train_ix, test_ix in cv_outer.split(X,y):# Now the nested cross validation begins
    print(i)
    # split data
    X_train = X.iloc[train_ix,:]
    X_test = X.iloc[test_ix,:]
    y_train = y.iloc[train_ix]
    y_test = y.iloc[test_ix]
    #----adversarial validation between x_train and x_test from cv outer splits
    train_adv_val=X_train
    test_adv_val=X_test
    #  y here is the index of belonging or not to the train set 
    X_adv=train_adv_val.append(test_adv_val)
    y_adv=[0]*len(train_adv_val)+[1]*len(test_adv_val)
    np.random.seed(1234)
    # cross validation
    cv_preds=cross_val_predict(clf_pipeline2,X_adv,y_adv,cv=5,n_jobs=-1,method="predict_proba")
    print(roc_auc_score(y_true=y_adv,y_score=cv_preds[:,1]))# 
    roc_auc_cv_outer_results.append(roc_auc_score(y_true=y_adv,y_score=cv_preds[:,1]))
    #--adversarial validation between x_train from cv outer splits and out_sample test set of the public leaderboard
    # create train and test sets
    train_adv_val_public_leaderboard=X_train
    test_adv_val_public_leaderboard=new_titanic_test_data
    # Union of X and y where y here is the index of belonging or not to the train set 
    X_adv_public_leaderboard=train_adv_val_public_leaderboard.append(test_adv_val_public_leaderboard)
    y_adv_public_leaderboard=[0]*len(train_adv_val_public_leaderboard)+[1]*len(test_adv_val_public_leaderboard)
    np.random.seed(1234)
    # cross validation

    cv_preds_public_leaderboard=cross_val_predict(clf_pipeline2,X_adv_public_leaderboard,
                                                   y_adv_public_leaderboard,cv=5,n_jobs=-1,method="predict_proba")
    #model_fit=model.fit(X,y)
    print(roc_auc_score(y_true=y_adv_public_leaderboard,
                        y_score=cv_preds_public_leaderboard[:,1]))#  0.67 ROC -AUC score for our predictions
    roc_auc_cv_results_public_leaderboard.append(roc_auc_score(y_true=y_adv_public_leaderboard,
                                                                y_score=cv_preds_public_leaderboard[:,1]))
    i=i+1

print('Pipeline 2: Roc-Auc average score between train and test sets of cv outer splits: %.3f (%.3f)' % (np.mean(roc_auc_cv_outer_results), 
                                         np.std(roc_auc_cv_outer_results))) #0.506 (0.046)
print('Pipeline 2: Roc-Auc average score between train from cv outer splits and public learderboard test sets: %.3f (%.3f)' % (np.mean(roc_auc_cv_results_public_leaderboard),
                                                  np.std(roc_auc_cv_results_public_leaderboard))) # 0.491 (0.008)

