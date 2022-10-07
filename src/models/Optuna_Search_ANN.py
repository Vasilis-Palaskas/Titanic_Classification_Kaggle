
""" Hyperparameter tuning using nested cross validation scheme through Optuna method
    (both hyperparameters and data processing pipelines)  using Keras Tensorflow Classifier.
"""
#%% Keras Classifier Instance
# Function to create model, required for KerasClassifier using prior-specified number of layers.
#def create_ann_model(optimizer='adam',neurons=32,dropout_rate=0.1,
#                 init_mode='uniform',activation="relu"):
#    	# create model
#	ann_model = Sequential()
#	ann_model.add(Dense(neurons, input_shape=(14,),kernel_initializer=init_mode, #14 will be the dimension of X
#                 activation=activation))
#	ann_model.add(Dense(neurons,  kernel_initializer=init_mode, activation=activation))
#	ann_model.add(Dropout(dropout_rate))
#	ann_model.add(Dense(1,kernel_initializer=init_mode, activation='sigmoid'))
#	# Compile model
#	ann_model.compile(loss='binary_crossentropy', optimizer=optimizer,
#                 metrics=['accuracy'])
#	return ann_model

# Function to create model, required for KerasClassifier. 
#  Convenient acronym ann ( Artificial Neural Networks ) to be used from now and on for the results related to the specific classifier

def create_ann_model(optimizer='adam',neurons=32,dropout_rate=0.1,
                 init_mode='uniform',activation="relu",layers=3):
    
    ann_model = Sequential()
    for i in range(1,layers):
        if i==1:
            ann_model.add(Dense(neurons,input_dim =14,kernel_initializer=init_mode,activation = activation))
        else:
            ann_model.add(Dense(neurons,activation = activation))
            
    ann_model.add(Dropout(dropout_rate))
    ann_model.add(Dense(1,kernel_initializer=init_mode,activation = 'sigmoid'))
    # Compile model
    ann_model.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
    return ann_model



# Data Processing Pipeline 1: Combine data processing pipelines (categorical+ordinal+numeric) Where the numeric features are standardized
pipeline1_classifiers = ColumnTransformer([("cat_pipeline", cat_pipeline_adv_val, cat_vars),
                                        ("ord_pipeline", ord_pipeline_adv_val, ord_vars),
                                       ("num_pipeline",num_pipeline1, num_vars)])


# Data Processing Pipeline 2: Combine data processing pipelines (categorical+ordinal+numeric) combine pipelines Where the numeric features are normalized
pipeline2_classifiers = ColumnTransformer([("cat_pipeline", cat_pipeline_adv_val, cat_vars),
                                        ("ord_pipeline", ord_pipeline_adv_val, ord_vars),
                                       ("num_pipeline",num_pipeline2, num_vars)])


# --- Define nested cross-validation scheme
cv_outer = StratifiedKFold(n_splits=10,shuffle=True, random_state=52)# 10, 52
cv_inner = StratifiedKFold(n_splits=6,shuffle=True, random_state=52) # Future consideration: RepeatedStratifiedKFold(n_repeats=3,n_splits=10,random_state=0)

#%% Nested cross validation block


#### Below, we initialize the corresponding lists including in each cv outer split (in total 10) some useful results

# out and in sample accuracies for cv outer splits
outer_results = list()
in_sample_acc_results = list()

# out and in sample f1 scores for cv outer splits
outer_results_f1 = list()
in_sample_f1_results = list()

# out and in sample predictions for cv outer splits
outer_yhat_optuna_ann =list()
outer_yhat_train_optuna_ann =list()

# best params and score for cv inner splits
inner_best_params=list()
inner_best_score=list()

# List including the predictions of the Competittion test dataset (public leaderboard
y_out_sample_test_results=list()


# Now the nested cross validation begins
i=0#counter
for train_ix, test_ix in cv_outer.split(X,y):
    i=i+1
    print(i)
    # split data
    X_train = X.iloc[train_ix,:]
    X_test = X.iloc[test_ix,:]
    y_train = y.iloc[train_ix]
    y_test = y.iloc[test_ix]

    #-------- Optuna Hyperparameters Optimisation method (ANN/Keras Classifier)
    np.random.seed(1234)
    class Objective:
        def __init__(self, early_stop):
            self.best_booster = None
            self._booster = None
            self.early_stop_rounds = early_stop# refers to the early stop in epochs
        def __call__(self, trial, X, y, cv, scoring):
            # -- Instantiate scaler
            pipelines=trial.suggest_categorical("pipelines",["pipeline1_stdscaler",
                                                             "pipeline2_normalizer"])
            # (b) Define your pipelines
            if pipelines == "pipeline1_stdscaler":
                pipelines = pipeline1_classifiers
            elif pipelines == "pipeline2_normalizer": 
                pipelines = pipeline2_classifiers
            # -- ann/Keras Classifier hyperparams distributions to search
            ann_params= {
                "neurons": trial.suggest_int("neurons", 96,208),
                "dropout_rate": trial.suggest_float("dropout_rate", 0,0.5),
                "optimizer": trial.suggest_categorical("optimizer", ["adam","rmsprop"]),
                "activation": trial.suggest_categorical("activation", ['relu','tanh']),
                "init_mode": trial.suggest_categorical("init_mode", ['glorot_normal', 'uniform']),
                'epochs':trial.suggest_int("epochs", 50,120),
                'class_weight': trial.suggest_categorical("class_weight", ['balanced',None]),
                "layers": trial.suggest_int("layers", 2,4)#in essence, they are 2 or 3 layers
                }
            # -- Define Estimator (ann/Keras Classifier)
            estimator=KerasClassifier(build_fn=create_ann_model,verbose=0,**ann_params)
            # -- Make a pipeline including data processing pipelines along with estimator 
            # -- so our hyperparams search to search which data processing pipelines and model hyperparameters provide
            # -- the best combination in cv inner splits
            pipeline = make_pipeline(pipelines,  estimator)
            self._booster =pipeline
            scores = cross_val_score(pipeline, X, y, cv=cv,scoring=scoring)
            accuracy =scores.mean()*100
            return accuracy
        def callback(self, study, trial):# provide the info that we want to observe and which model version to fit as the best one based on hyperparams search
            if study.best_trial == trial:
                self.best_booster = self._booster
                
    # run objective object
    early_stop_rounds=15
    objective = Objective(early_stop_rounds)
    objective.early_stop_rounds#=early_stop_rounds
    study = optuna.create_study(direction="maximize")
    # Pass additional arguments inside another function
    func = lambda trial: objective(trial, X_train, y_train, cv=cv_inner, scoring="accuracy")
    n_trials=40# someone could add more and more depending on the resources he uses (for 10*6 splits, he took 3-4 hours)
    #maximum_time = 0.2*60*60# seconds
    study.optimize(func, n_trials=n_trials, callbacks=[objective.callback])
    
    #---Visualisation
    #plot_optimization_history(study);
    #plot_param_importances(study);
    
    # get the best performing model fit by optuna-based search  in inner-k-fold splits of the train dataset from outer split.
    best_model = objective.best_booster
    best_parameters =study.best_params
    # evaluate model on the hold out dataset of each cv outer split
    best_model.fit(X_train, y_train)
    yhat = best_model.predict(X_test)
    acc = accuracy_score(y_test, yhat).round(2)*100 
    f1_out=f1_score(y_test, yhat).round(2)*100  # 
    # evaluate model on the hold in dataset  of each cv outer split
    yhat_train = best_model.predict(X_train)
    in_sample_acc=accuracy_score(y_train, yhat_train).round(2)*100 
    f1_in=f1_score(y_train, yhat_train).round(2)*100  # 
    
    # store the results
    outer_results.append(acc)
    outer_results_f1.append(f1_out)
    outer_yhat_optuna_ann.append(yhat)
    outer_yhat_train_optuna_ann.append(yhat_train)
    
    # store the inner results
    in_sample_acc_results.append(in_sample_acc)
    in_sample_f1_results.append(f1_in)
    inner_best_params.append(best_parameters)
    #inner_best_score.append(grid_hyperparams_inner_result.best_score_)
    
    # Public Leaderboard test dataset predictions using nested-cross validation models' predictions
    y_out_sample_test = best_model.predict(new_titanic_test_data)
    y_out_sample_test_results.append(y_out_sample_test)
    
    # report in-sample and out-sample progress per cv outer split
    print('>out-sample acc=%.3f, f1=%.3f' % (acc,f1_out))
    print('>in-sample acc=%.3f,  f1=%.3f' % (in_sample_acc, f1_in))
    # report progress of accuracy scores along with the best parameters after each cv outer split run
    print('>acc=%.3f, cfg=%s' % (acc,  best_parameters))

#%% Output after Hyperparameter tuning of Optuna search and predictions
#--- summarize the estimated performance of the model
print('Out-Sample Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results))) #Out-Sample Accuracy:81.500 (3.202)-- 80.600 (3.747)
print('Out-Sample F1 score: %.3f (%.3f)' % (np.mean(outer_results_f1), np.std(outer_results_f1))) #F1:74.000 (3.950)---73.000 (4.669)

print('In-Sample Accuracy: %.3f (%.3f)' % (np.mean(in_sample_acc_results), np.std(in_sample_acc_results))) #Accuracy: 85.900 (0.700)---86.300 (1.187)
print('In-Sample F1 score: %.3f (%.3f)' % (np.mean(in_sample_f1_results), np.std(in_sample_f1_results))) #F1: 80.100 (1.578)----80.700 (2.238)

#%% Output of Baseline ANN model and predictions (the below hyperparams were suggested as an initial approach by the book MLE by Andriy Burkov.)
tf.random.set_seed(42)# set seed for reproducibility
tf_baseline = KerasClassifier(build_fn=create_ann_model,
                                   epochs=75,  verbose=0,dropout_rate=0.0,
                                  init_mode='glorot_normal',neurons=128,activation='relu',
                                  optimizer='adam',layers=3)
# Baseline Classifier along with the data processing pipeline 1 defined above
tf_baseline_pipeline1 = Pipeline(steps=[
    ('col_trans', pipeline1_classifiers),
    ('model',tf_baseline)])

baseline_scores = cross_val_score(tf_baseline_pipeline1,X,y,
                                  cv=cv_outer,scoring="accuracy")
baseline_accuracy =baseline_scores.mean()*100
baseline_accuracy
print("%0.2f accuracy with a standard deviation of %0.2f" % (baseline_scores.mean()*100, 
                                                             baseline_scores.std()*100))#81.60 (4.71)

#%%%% Save model training and model prediction results both inner and outer cv splits 

#---Choose working directory
directory='C:/Users/vasileios palaskas/Documents/GitHub/Titanic_Classification_Kaggle/models/ANN_KerasClassifier'
logger.info('Define the directory of your saved model objects (folder Titanic_Classification_Kaggle/models/ANN_KerasClassifier')
os.chdir(directory)

# Write-Save objects related to the Nested Cross-validation prodedure implemented through ann/Keras Classifier

# CV outer splits: Out-sample results (accuracies)
with open("outer_results_optuna_ann", "wb") as outer_results_fp:   #Pickling
 pickle.dump(outer_results, outer_results_fp)

# CV outer splits: In-sample results (accuracies)
with open("in_sample_acc_results_optuna_ann", "wb") as in_sample_acc_results_fp:   #Pickling
  pickle.dump(in_sample_acc_results, in_sample_acc_results_fp)

# CV outer splits: Out-sample predictions of the response 
with open("outer_yhat_optuna_ann", "wb") as outer_yhat_optuna_ann_fp:   #Pickling
  pickle.dump(outer_yhat_optuna_ann, outer_yhat_optuna_ann_fp)
  
# CV outer splits: Public Leaderboard test dataset predictions of the response 
with open("y_out_sample_test_results_optuna_ann", "wb") as y_out_sample_test_results_fp:   # predictions 
    pickle.dump(y_out_sample_test_results, y_out_sample_test_results_fp)
    
# CV ínner splits: Best parameters emerged during cross-validation in inner splits (6 in total) 
with open("inner_best_params_optuna_ann", "wb") as inner_best_params_optuna_ann:   # best params in each outer cross-validation trial
    pickle.dump(inner_best_params, inner_best_params_optuna_ann)

# Load objects


# CV outer splits: Out-sample results (accuracies)
with open("outer_results_optuna_ann", "rb") as outer_results_fp:   # Unpickling
 outer_results_optuna_ann = pickle.load(outer_results_fp)
 
# CV outer splits: In-sample results (accuracies)
with open("in_sample_acc_results_optuna_ann", "rb") as in_sample_acc_results_fp:   # Unpickling
  in_sample_acc_results_optuna_ann = pickle.load(in_sample_acc_results_fp)
  
# CV outer splits: Out-sample predictions of the response    
with open("outer_yhat_optuna_ann", "rb") as outer_yhat_optuna_ann_fp:   # Unpickling
        yhat_optuna_ann_list = pickle.load(outer_yhat_optuna_ann_fp)  

# CV outer splits: Public Leaderboard test dataset predictions of the response         
with open("y_out_sample_test_results_optuna_ann", "rb") as y_out_sample_test_results_fp:   # predictions 
   y_out_sample_test_results_optuna_ann_list = pickle.load(y_out_sample_test_results_fp) 

# CV ínner splits: Best parameters emerged during cross-validation in inner splits (6 in total)    
with open("inner_best_params_optuna_ann", "rb") as inner_best_params_optuna_ann:   # best params in each outer cross-validation trial
   inner_best_params = pickle.load(inner_best_params_optuna_ann) 

  

