
#%% Preparation for the hyperparams search and cross validation scheme

model = XGBClassifier(eval_metric="logloss")# Initial XGB Classifier

# ---- Data processing pipelines

coltrans3 = ColumnTransformer([("cat_pipeline", cat_pipeline, cat_vars),
                               ("ord_pipeline", ord_pipeline, ord_vars)],
                                  remainder="passthrough")
# Final candidate pipelines: Combination of processes into single pipelines


num_pipeline1= Pipeline( [  ('scaler',  StandardScaler()),
                            ('imputer', KNNImputer(n_neighbors=10))] 
                        )

# combine pipelines
pipeline1 = ColumnTransformer([("cat_pipeline", cat_pipeline, cat_vars),
                                        ("ord_pipeline", ord_pipeline, ord_vars),
                                       ("num_pipeline",num_pipeline1, num_vars)])

clf_pipeline1 = Pipeline(steps=[
    ('col_trans', pipeline1),
    ('model',model)])

#--Pipeline 2

num_pipeline2 = Pipeline( [('normalizer',  MinMaxScaler()),
                            ('imputer', KNNImputer(n_neighbors=10))] )
# combine pipelines
pipeline2 = ColumnTransformer([("cat_pipeline", cat_pipeline, cat_vars),
                                        ("ord_pipeline", ord_pipeline, ord_vars),
                                       ("num_pipeline",num_pipeline2, num_vars)])

pipeline3 =  Pipeline([('column_transformer', coltrans3),
                    ('scaler', StandardScaler()), ('imputer', KNNImputer(n_neighbors=10))])

# --- Define nested cross-validation scheme
cv_outer = StratifiedKFold(n_splits=10,shuffle=True, random_state=52)# 10, 52
cv_inner = StratifiedKFold(n_splits=6,shuffle=True, random_state=52) # Future consideration: RepeatedStratifiedKFold(n_repeats=3,n_splits=10,random_state=0)

#%% Nested cross validation

#--- Enumerate splits and initialise a vector with results
# out and in sample accuracies for cv outer splits
outer_results = list()
in_sample_acc_results = list()
# out and in sample f1 scores for cv outer splits

outer_results_f1 = list()
in_sample_f1_results = list()
# out and in sample predictions for cv outer splits

outer_yhat_optuna_xgb =list()
outer_yhat_train_optuna_xgb =list()

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

    #-------- Optuna Hyperparameters Optimisation method (XGBClassifier)
    np.random.seed(1234)
    class Objective:
        def __init__(self, early_stop):
            self.best_booster = None
            self._booster = None
            self.early_stop_rounds = early_stop
        def __call__(self, trial, X, y, cv, scoring):
            # -- Instantiate scaler
            pipelines=trial.suggest_categorical("pipelines",["pipeline1_stdscaler","pipeline2_normalizer"])
            # (b) Define your pipelines
            if pipelines == "pipeline1_stdscaler":
                pipelines = pipeline1
            elif pipelines == "pipeline2_normalizer": 
                pipelines = pipeline2
            # -- XGB Classifier hyperparams
            xgb_params= {
                "n_estimators": trial.suggest_int("n_estimators", 50,1990, step=20),
                "learning_rate": trial.suggest_float("learning_rate", 0.01,1.0),
                "max_depth": trial.suggest_int("max_depth", 2, 92),
                "subsample": trial.suggest_float("subsample", 0.3,1, step=0.1),
                #"max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3,1, step=0.1),
                # L2 regularization
                'reg_lambda':trial.suggest_float("reg_lambda", 0.01,100),
                # L1 regularisation
                'reg_alpha':trial.suggest_float("reg_alpha", 0.01,100),
                "random_state": 52,
                "n_jobs": -1,
                "eval_metric":"logloss",
                "objective":'binary:logistic',
                "use_label_encoder":False
                }
            # -- Estimator (XGB Classifier initial version)
            estimator=XGBClassifier(**xgb_params)
            # -- Make a pipeline
            pipeline = make_pipeline(pipelines,  estimator)
            self._booster =pipeline
            scores = cross_val_score(pipeline, X, y, cv=cv,scoring=scoring)
            accuracy =scores.mean()*100
            return accuracy
        def callback(self, study, trial):
            if study.best_trial == trial:
                self.best_booster = self._booster

    # run objective object
    early_stop_rounds=50
    objective = Objective( early_stop_rounds)
    study = optuna.create_study(direction="maximize")
    # Pass additional arguments inside another function
    func = lambda trial: objective(trial, X_train, y_train, cv=cv_inner, scoring="f1")
    n_trials=150
    #maximum_time = 0.2*60*60# seconds
    study.optimize(func, n_trials=n_trials, callbacks=[objective.callback])
    
    #---Visualisation
    #plot_optimization_history(study);
    #plot_param_importances(study);
    # get the best performing model fit by grid-searching in inner-k-fold splits of the train dataset from outer split.
    best_model = objective.best_booster
    best_parameters =study.best_params
    # evaluate model on the hold out dataset
    best_model.fit(X_train, y_train)
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
    outer_yhat_optuna_xgb.append(yhat)
    outer_yhat_train_optuna_xgb.append(yhat_train)
    
    in_sample_acc_results.append(in_sample_acc)
    in_sample_f1_results.append(f1_in)
    # store the inner results
    inner_best_params.append(best_parameters)
    #inner_best_score.append(grid_hyperparams_inner_result.best_score_)
    # test predictions for the competitions using nested-cross val runs
    y_out_sample_test = best_model.predict(new_titanic_test_data)
    y_out_sample_test_results.append(y_out_sample_test)
    # report progress per cv outer split
    print('>out-sample acc=%.3f, f1=%.3f,  cfg=%s' % (acc,f1_out, best_parameters))
    print('>in-sample acc=%.3f,  f1=%.3f' % (in_sample_acc, f1_in))

#%% Output of Optuna search and predictions
#--- summarize the estimated performance of the model
print('Out-Sample Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results))) #Out-Sample Accuracy: 82.600 (2.835)
print('Out-Sample F1 score: %.3f (%.3f)' % (np.mean(outer_results_f1), np.std(outer_results_f1))) #F1:76.000 (4.050)

print('In-Sample Accuracy: %.3f (%.3f)' % (np.mean(in_sample_acc_results), np.std(in_sample_acc_results))) #Accuracy:89.900 (1.221)
print('In-Sample F1 score: %.3f (%.3f)' % (np.mean(in_sample_f1_results), np.std(in_sample_f1_results))) #F1:86.600 (1.562)

#%%%% Save model training and model prediction results both inner and outer cv splits 
with open("outer_results_optuna_xgb", "wb") as outer_results_fp:   #Pickling
 pickle.dump(outer_results, outer_results_fp)

with open("outer_results_optuna_xgb", "rb") as outer_results_fp:   # Unpickling
 outer_results_optuna_xgb = pickle.load(outer_results_fp)
 
#
 with open("in_sample_acc_results_optuna_xgb", "wb") as in_sample_acc_results_fp:   #Pickling
  pickle.dump(in_sample_acc_results, in_sample_acc_results_fp)

 with open("in_sample_acc_results_optuna_xgb", "rb") as in_sample_acc_results_fp:   # Unpickling
  in_sample_acc_results_optuna_xgb = pickle.load(in_sample_acc_results_fp)

 #
 with open("outer_yhat_optuna_xgb", "wb") as outer_yhat_optuna_xgb_fp:   #Pickling
  pickle.dump(outer_yhat_optuna_xgb, outer_yhat_optuna_xgb_fp)
       
with open("outer_yhat_optuna_xgb", "rb") as outer_yhat_optuna_xgb_fp:   # Unpickling
        yhat_optuna_xgb_list = pickle.load(outer_yhat_optuna_xgb_fp)
#%%%% Store our predictions (saved in models/XGB_Predictions)
with open("y_out_sample_test_results_optuna_xgb", "wb") as y_out_sample_test_results_fp:   #Pickling
    pickle.dump(y_out_sample_test_results, y_out_sample_test_results_fp)

with open("y_out_sample_test_results_optuna_xgb", "rb") as y_out_sample_test_results_fp:   # Unpickling
   y_out_sample_test_results_optuna_xgb_list = pickle.load(y_out_sample_test_results_fp) 

  
#%% Code for objective class+additional commands for single executions of this class (visualisation, convergence)
# Create the objective function that we want to optimise for


# def objective(trial, X, y, cv, scoring):
#     # -- Instantiate scaler

#     pipelines=trial.suggest_categorical("pipelines",["pipeline1_stdscaler","pipeline2_normalizer"])
#     # (b) Define your pipelines
#     if pipelines == "pipeline1_stdscaler":
#         pipelines = pipeline1
#     elif pipelines == "pipeline2_normalizer": 
#         pipelines = pipeline2
#     # -- XGB Classifier hyperparams
#     xgb_params= {
#         "n_estimators": trial.suggest_int("n_estimators", 50,2000, step=20),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01,1.0),
#         "max_depth": trial.suggest_int("max_depth", 3, 9),
#         "subsample": trial.suggest_float("subsample", 0.3,1, step=0.1),
#         #"max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"]),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3,1, step=0.1),
#         # L2 regularization
#         'reg_lambda':trial.suggest_float("reg_lambda", 0.01,100),
#         # L1 regularisation
#         'reg_alpha':trial.suggest_float("reg_alpha", 0.01,100),
#         "random_state": 52,
#         "n_jobs": -1,
#         "eval_metric":"logloss",
#         "objective":'binary:logistic',
#         "use_label_encoder":False
#         }
#     # -- Estimator (XGB Classifier initial version)
#     estimator=XGBClassifier(**xgb_params)
#     global estimator
#     # -- Make a pipeline
#     pipeline = make_pipeline(pipelines,  estimator)
#     scores = cross_val_score(pipeline, X, y, cv=cv,scoring=scoring)
#     accuracy =scores.mean()*100
#     return accuracy


# def callback(study, trial):
#     global best_booster
#     if study.best_trial == trial:
#         best_booster = estimator


# # Initiate the object of study (optuna optimiser)
# study = optuna.create_study(direction="maximize") # maximise the score during tuning
# # Pass additional arguments inside another function
# func = lambda trial: objective(trial, X, y, cv=cv_inner, scoring="accuracy")
# warnings.filterwarnings("ignore")
# # Ignore warnings
# warnings.filterwarnings("ignore", category=DataConversionWarning)#
# warnings.filterwarnings("ignore", category=DeprecationWarning)#elimi
# # Start optimizing
# study.optimize(func, n_trials=50)

# print(f"Optimized Accuracy : {study.best_value:.5f}")#82.04
# # Visualisation with the results
# from optuna.visualization.matplotlib import plot_optimization_history,plot_param_importances

# plot_optimization_history(study);
# plot_param_importances(study);
