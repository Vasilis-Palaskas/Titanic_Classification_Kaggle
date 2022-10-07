
import datetime
import keras_tuner
import scikeras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
# ANN via tensorflow
tf.random.set_seed(42)
ann_model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=128,   
                          kernel_initializer=initializers.RandomNormal(stddev=0.01),#tf.keras.initializers.GlorotNormal()
                          bias_initializer=initializers.Zeros(), 
                          activation='relu'),
    tf.keras.layers.Dense(units=128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

ann_model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(name="Adam"),#RMSprop (Root Mean Squared Propagation), SGD
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
            ]
                )
feature_cols = tf.contrib.learn.infer_real_valued_columns_from_input(X)
X_cat_ord_pipeline_with_ordinal=pd.DataFrame(pipeline1.fit_transform(X))
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)

history = ann_model.fit(X_cat_ord_pipeline_with_ordinal, y, epochs=100, callbacks=[callback])
history = ann_model.fit(X_cat_ord_pipeline_with_ordinal, y, epochs=100)
#Returns the best hyperparameters, as determined by the objective.
#This method can be used to reinstantiate the (untrained) best model found during the search process.
best_hp = Tuner.get_best_hyperparameters(num_trials=1)
model = tuner.hypermodel.build(best_hp)

tf.keras.utils.plot_model(ann_model,
    to_file="model.png")

# Visualize the performance of ann classifier

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['figure.figsize'] = (18, 8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False


plt.plot(
    np.arange(1, 101), 
    history.history['loss'], label='Loss'
)
plt.plot(
    np.arange(1, 101), 
    history.history['accuracy'], label='Accuracy'
)
plt.plot(
    np.arange(1, 101), 
    history.history['precision'], label='Precision'
)
plt.plot(
    np.arange(1, 101), 
    history.history['recall'], label='Recall'
)
plt.title('Evaluation metrics', size=20)
plt.xlabel('Epoch', size=14)
plt.legend();





#%% Preparation for the hyperparams search and cross validation scheme
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasClassifier
# ---- Data processing pipelines& Classifier combination
tf.random.set_seed(42)
# Function to create model, required for KerasClassifier
def create_model(optimizer='adam',neurons=32,dropout_rate=0.46722513265844545,
                 init_mode='uniform',activation="relu"):
	# create model
	ann_model = Sequential()
	ann_model.add(Dense(neurons, input_shape=(14,),kernel_initializer=init_mode, 
                 activation=activation))
	ann_model.add(Dense(neurons,  kernel_initializer=init_mode, activation=activation))
	ann_model.add(Dropout(dropout_rate))
	ann_model.add(Dense(1,kernel_initializer=init_mode, activation='sigmoid'))
	# Compile model
	ann_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return ann_model


tf_keras_initial = KerasClassifier(build_fn=create_model,
                                   epochs=50, batch_size=10, verbose=0,dropout_rate=0.46722513265844545,
                                   init_mode='glorot_normal',neurons=128,activation='tanh')
#history = tf_keras_initial.fit(pipeline_1_2.fit_transform(X),y)

# Data processing pipelines to be compared during nested cross-val.
num_pipeline_1_2=Pipeline( [('scaler',  StandardScaler()),
                            ('normaliser', MinMaxScaler()),
                            ('imputer', KNNImputer(n_neighbors=10))] )
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent') ),
                            ('encoder', OneHotEncoder(handle_unknown='ignore',drop = 'first'))])

ord_pipeline=Pipeline([('imputer', SimpleImputer(strategy='most_frequent') ),
                            ('encoder', OrdinalEncoder())]
                      )
pipeline_1_2 = ColumnTransformer([("cat_pipeline", cat_pipeline, cat_vars),                                   
                                 ("num_pipeline",num_pipeline_1_2, num_vars)])# std scaler or normaliser for numeric
# Combin pipeline+classifier
tf_keras_clf_pipeline_1_2 = Pipeline(steps=[
    ('col_trans', pipeline_1_2),
    ('model',tf_keras_initial) ])

# --- Define nested cross-validation scheme
cv_outer = StratifiedKFold(n_splits=10,shuffle=True, random_state=52)
cv_inner = StratifiedKFold(n_splits=6,shuffle=True, random_state=52) # Future consideration: RepeatedStratifiedKFold(n_repeats=3,n_splits=10,random_state=0)
                        
# Define parameter space for search depending on the specific classifier
tf_keras_params= {
    # Number of boosted trees to fit
    #"model__optimizer":Categorical(["rmsprop","adam","sgd"]),
    #"model__neurons": Integer( 32,256),
    #"model__dropout_rate": Real(0,0.1,'uniform'),
    "model__activation": Categorical( [ 'relu','tanh']),
    #"model__init_mode" : Categorical( ['glorot_normal', 'glorot_uniform'])#,'uniform'
    #"model__colsample_bytree": Real(0.3,1,'uniform'),
    # L2 regularization
    #'model__reg_lambda':Real(1e-2,100,'log-uniform'),
    # L1 regularisation
    #'model__reg_alpha':Real(1e-2,100,'log-uniform')
        }

#RMSprop (Root Mean Squared Propagation), SGD
grid_tf_keras_params = [{**{'col_trans__num_pipeline__normaliser': ['passthrough']}, **tf_keras_params},
                    {**{'col_trans__num_pipeline__scaler': ['passthrough']}, **tf_keras_params}]


#--- Enumerate splits and initialise a vector with results
# out and in sample accuracies for cv outer splits
outer_results = list()
in_sample_acc_results = list()
# out and in sample f1 scores for cv outer splits

outer_results_f1 = list()
in_sample_f1_results = list()
# out and in sample predictions for cv outer splits

outer_yhat_bayessearch_ann =list()
outer_yhat_train_bayessearch_ann =list()

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
    tf.random.set_seed(42)

    opt_inner =BayesSearchCV(estimator=tf_keras_clf_pipeline_1_2,search_spaces=grid_tf_keras_params,
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


    outer_yhat_bayessearch_ann.append(yhat)
    outer_yhat_train_bayessearch_ann.append(yhat_train)
    
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
with open("outer_results_bayessearch_ann", "wb") as outer_results_fp:   #Pickling
 pickle.dump(outer_results, outer_results_fp)

with open("outer_results_bayessearch_ann", "rb") as outer_results_fp:   # Unpickling
 outer_results_bayessearch_ann = pickle.load(outer_results_fp)
#
 with open("in_sample_acc_results_bayessearch_ann", "wb") as in_sample_acc_results_fp:   #Pickling
  pickle.dump(in_sample_acc_results, in_sample_acc_results_fp)

 with open("in_sample_acc_results_bayessearch_ann", "rb") as in_sample_acc_results_fp:   # Unpickling
  in_sample_acc_results_bayessearch_ann = pickle.load(in_sample_acc_results_fp)
 #
 with open("outer_yhat_bayessearch_ann", "wb") as outer_yhat_bayessearch_ann_fp:   #Pickling
  pickle.dump(outer_yhat_bayessearch_ann, outer_yhat_bayessearch_ann_fp)
with open("outer_yhat_bayessearch_ann", "rb") as outer_yhat_bayessearch_ann_fp:   # Unpickling
        yhat_bayessearch_ann_list = pickle.load(outer_yhat_bayessearch_ann_fp)
#%%%% Store our predictions (saved in models/RF_Predictions)

with open("y_out_sample_test_results_bayessearch_ann", "wb") as y_out_sample_test_results_fp:   #Pickling
 pickle.dump(y_out_sample_test_results, y_out_sample_test_results_fp)

with open("y_out_sample_test_results_bayessearch_ann", "rb") as y_out_sample_test_results_fp:   # Unpickling
 y_out_sample_test_results_bayessearch_ann_list = pickle.load(y_out_sample_test_results_fp) 

         
 
