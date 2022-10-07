
""" Hyperparameter tuning using nested cross validation scheme through Bayes Search method
    (both hyperparameters and data processing pipelines)  using Random Forest Classifier.

"""

#%% Keras Classifier Instance
# Function to create model, required for KerasClassifier
def create_ann_model(optimizer='adam',neurons=32,dropout_rate=0.1,
                 init_mode='uniform',activation="relu"):
    	# create model
	ann_model = Sequential()
	ann_model.add(Dense(neurons, input_shape=(14,),kernel_initializer=init_mode, #14 will be the dimension of X
                 activation=activation))
	ann_model.add(Dropout(dropout_rate))
	ann_model.add(Dense(1,kernel_initializer=init_mode, activation='sigmoid'))
	# Compile model
	ann_model.compile(loss='binary_crossentropy', optimizer=optimizer,
                 metrics=['accuracy'])
	return ann_model
# Create a KerasClassifier object using some specific model-parameters
#tf.random.set_seed(42)# set seed for reproducibility
#tf_keras_initial = KerasClassifier(build_fn=create_ann_model,
#                                   epochs=50, batch_size=10, verbose=0,dropout_rate=0.46722513265844545,
#                                   init_mode='glorot_normal',neurons=64,activation='tanh')

# combine pipelines Where the numeric features are standardized
pipeline_1_2 = ColumnTransformer([("cat_pipeline",cat_pipeline_adv_val, cat_vars),                                   
                                 ("num_pipeline",num_pipeline_1_2, num_vars)])# std scaler or normaliser for numeric

# Baseline ann model
tf.random.set_seed(42)# set seed for reproducibility
tf_baseline = KerasClassifier(build_fn=create_ann_model,
                                   epochs=75,  verbose=0,dropout_rate=0.0,
                                  init_mode='glorot_normal',neurons=128,activation='relu',
                                  optimizer='adam')
# Baseline Classifier along with the data processing pipeline 1 defined above
tf_baseline_pipeline1_2 = Pipeline(steps=[
    ('col_trans', pipeline_1_2),
    ('model',tf_baseline) ])
# --- Define nested cross-validation scheme
cv_outer = StratifiedKFold(n_splits=10,shuffle=True, random_state=52)# 10, 52
cv_inner = StratifiedKFold(n_splits=6,shuffle=True, random_state=52) # Future consideration: RepeatedStratifiedKFold(n_repeats=3,n_splits=10,random_state=0)
                    
# Define parameter space for search depending on the specific classifier

ann_params= {
    # Number of boosted trees to fit
    'model__neurons':Integer( 96,192),
    "model__dropout_rate":  Real(0,0.5,'uniform'),
    "model__epochs": Integer( 50,100),
    #"model__max_depth": Integer(2, 92),
    #"model__subsample": Real(0.3,1,'uniform'),
    "model__init_mode": Categorical( ['glorot_normal', 'uniform']),
    "model__activation" : Categorical( ['relu'])
        }


grid_ann_params = [{**{'col_trans__num_pipeline__normaliser': ['passthrough']}, **ann_params},
                    {**{'col_trans__num_pipeline__scaler': ['passthrough']}, **ann_params}]


#%% Nested cross validation

#--- Enumerate splits and initialise a vector with results
# out and in sample accuracies for cv outer splits
outer_results = list()
in_sample_acc_results = list()
# out and in sample f1 scores for cv outer splits

outer_results_f1 = list()
in_sample_f1_results = list()
# out and in sample predictions for cv outer splits

outer_yhat_bayessearach_1_layer =list()
outer_yhat_train_bayessearach_1_layer =list()

# best params and score for cv inner splits

inner_best_params=list()
inner_best_score=list()
#---Competittion test dataset predictions
y_out_sample_test_results=list()
i=0
# Now the nested cross validation begins
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

    opt_inner =BayesSearchCV(estimator=tf_baseline_pipeline1_2,search_spaces=grid_ann_params,
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

    outer_yhat_bayessearach_1_layer.append(yhat)
    outer_yhat_train_bayessearach_1_layer.append(yhat_train)
    
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
#%% Output of Baseline ANN model and predictions
tf.random.set_seed(42)# set seed for reproducibility
tf_baseline = KerasClassifier(build_fn=create_ann_model,
                                   epochs=75,  verbose=0,dropout_rate=0.0,
                                  init_mode='glorot_normal',neurons=128,activation='relu',
                                  optimizer='adam')
# Baseline Classifier along with the data processing pipeline 1 defined above
tf_baseline_pipeline1 = Pipeline(steps=[
    ('col_trans', pipeline2_classifiers),
    ('model',tf_baseline)])

baseline_scores = cross_val_score(tf_baseline_pipeline1,X,y,
                                  cv=cv_outer,scoring="accuracy")
baseline_accuracy =baseline_scores.mean()*100
baseline_accuracy
print("%0.2f accuracy with a standard deviation of %0.2f" % (baseline_scores.mean(), 
                                                             baseline_scores.std()))#(normalizer scaler) 0.79 accuracy with a standard deviation of 0.04
# normal scaler: 0.80 accuracy with a standard deviation of 0.04
# summarize the estimated performance of the model
print('Out-Sample Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results))) #Accuracy:Out-Sample Accuracy: 80.700 (3.195)
Out-Sample F1 score: 72.500 (4.129)
print('Out-Sample F1 score: %.3f (%.3f)' % (np.mean(outer_results_f1), np.std(outer_results_f1))) #F1:72.500 (4.129)

print('In-Sample Accuracy: %.3f (%.3f)' % (np.mean(in_sample_acc_results), np.std(in_sample_acc_results))) #Accuracy:  85.100 (1.044)
print('In-Sample F1 score: %.3f (%.3f)' % (np.mean(in_sample_f1_results), np.std(in_sample_f1_results))) #F1:78.500 (2.012)

#%%%% Save model training and model prediction results both inner and outer cv splits 
with open("outer_results_bayessearach_1_layer", "wb") as outer_results_fp:   #Pickling
 pickle.dump(outer_results, outer_results_fp)

with open("outer_results_bayessearach_1_layer", "rb") as outer_results_fp:   # Unpickling
 outer_results_bayessearach_1_layer = pickle.load(outer_results_fp)
#
 with open("in_sample_acc_results_bayessearach_1_layer", "wb") as in_sample_acc_results_fp:   #Pickling
  pickle.dump(in_sample_acc_results, in_sample_acc_results_fp)

 with open("in_sample_acc_results_bayessearach_1_layer", "rb") as in_sample_acc_results_fp:   # Unpickling
  in_sample_acc_results_bayessearach_1_layer = pickle.load(in_sample_acc_results_fp)
 #
 with open("outer_yhat_bayessearach_1_layer", "wb") as outer_yhat_bayessearach_1_layer_fp:   #Pickling
  pickle.dump(outer_yhat_bayessearach_1_layer, outer_yhat_bayessearach_1_layer_fp)
with open("outer_yhat_bayessearach_1_layer", "rb") as outer_yhat_bayessearach_1_layer_fp:   # Unpickling
        yhat_bayessearach_1_layer_list = pickle.load(outer_yhat_bayessearach_1_layer_fp)
#%%%% Store our predictions (saved in models/RF_Predictions)

with open("y_out_sample_test_results_bayessearach_1_layer", "wb") as y_out_sample_test_results_fp:   #Pickling
 pickle.dump(y_out_sample_test_results, y_out_sample_test_results_fp)

with open("y_out_sample_test_results_bayessearach_1_layer", "rb") as y_out_sample_test_results_fp:   # Unpickling
 y_out_sample_test_results_bayessearach_1_layer_list = pickle.load(y_out_sample_test_results_fp) 

         
 #%%%% Store the best params in each outer cross-validation trial
 with open("inner_best_params_bayessearach_1_layer", "wb") as inner_best_params_bayessearach_1_layer:   #Pickling
     pickle.dump(inner_best_params, inner_best_params_bayessearach_1_layer)

 with open("inner_best_params_bayessearach_1_layer", "rb") as inner_best_params_bayessearach_1_layer:   # Unpickling
    inner_best_params = pickle.load(inner_best_params_bayessearach_1_layer) 

   
