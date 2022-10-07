#% Load saved objects from Python scripts: Optuna_Search_XGB.py and Bayes_Search_RF.py
 # to use them as ingredients to be ensembled

#---Choose working directory (ann/Keras Classifier)
#directory='C:/Users/vasileios palaskas/Documents/GitHub/Titanic_Classification_Kaggle/models/ANN_KerasClassifier'
#logger.info('Define the directory of your saved model objects (folder Titanic_Classification_Kaggle/models/ANN_KerasClassifier')
#os.chdir(directory)


# CV outer splits: Out-sample results (accuracies)
#with open("outer_results_optuna_ann", "rb") as outer_results_fp:   # Unpickling
# outer_results_optuna_ann = pickle.load(outer_results_fp)
 
# CV outer splits: In-sample results (accuracies)
#with open("in_sample_acc_results_optuna_ann", "rb") as in_sample_acc_results_fp:   # Unpickling
#  in_sample_acc_results_optuna_ann = pickle.load(in_sample_acc_results_fp)
  
# CV outer splits: Out-sample predictions of the response    
#with open("outer_yhat_optuna_ann", "rb") as outer_yhat_optuna_ann_fp:   # Unpickling
#        yhat_optuna_ann_list = pickle.load(outer_yhat_optuna_ann_fp)  

# CV outer splits: Public Leaderboard test dataset predictions of the response         
#with open("y_out_sample_test_results_optuna_ann", "rb") as y_out_sample_test_results_fp:   # predictions 
#   y_out_sample_test_results_optuna_ann_list = pickle.load(y_out_sample_test_results_fp) 

# CV ínner splits: Best parameters emerged during cross-validation in inner splits (6 in total)    
#with open("inner_best_params_optuna_ann", "rb") as inner_best_params_optuna_ann:   # best params in each outer cross-validation trial
#   inner_best_params = pickle.load(inner_best_params_optuna_ann) 

##.... XGB....

##.... RF....


#%% Outer Splits cross validation

#-- Optuna XGB related lists to be initialised for each cv outer split
outer_yhat_optuna_xgb =list()
outer_results_optuna_xgb = list()
outer_results_f1_optuna_xgb = list()
y_out_sample_test_results_optuna_xgb=list()
#-- Bayesearch Random Forest related lists to be initialised for each cv outer split
outer_yhat_bayessearach_rnf =list()
outer_results_bayessearach_rnf = list()
outer_results_f1_bayessearach_rnf = list()
y_out_sample_test_results_bayessearach_rnf=list()
#-- Bayesearch Random Forest related lists to be initialised for each cv outer split
outer_yhat_optuna_ann =list()
outer_results_optuna_ann = list()
outer_results_f1_optuna_ann = list()
y_out_sample_test_results_optuna_ann=list()

#-- Ensembled models related lists to be initialised for each cv outer split
outer_yhat_ensembled=list()
outer_results_ensembled = list()
outer_results_f1_ensembled = list()
y_out_sample_test_results_ensembled=list()


i=0# counting the cv outer splits
# Now the nested cross validation begins
for train_ix, test_ix in cv_outer.split(X,y):
    #i=i+1
    print(i+1)

    #---- split data
    X_train = X.iloc[train_ix,:]
    X_test = X.iloc[test_ix,:]
    y_train = y.iloc[train_ix]
    y_test = y.iloc[test_ix]
    
    # The corresponding predictions for the test dataset of each cv outer split from each fitted model  in each cv outer split
    yhat_optuna_xgb=yhat_optuna_xgb_list[i]# 
    yhat_bayessearach_rnf=yhat_bayessearach_rnf_list[i]#
    yhat_optuna_ann=np.array(yhat_optuna_ann_list[i] )#
    yhat_optuna_ann=np.concatenate( yhat_optuna_ann, axis=0 )# convert an array including multiple-arrays to a single array
   
    # The corresponding predictions for the public leaderboard test dataset from each fitted model in each cv outer split
    y_out_sample_test_results_optuna_xgb=y_out_sample_test_results_optuna_xgb_list[i]
    y_out_sample_test_results_bayessearach_rnf=y_out_sample_test_results_bayessearach_rnf_list[i] 
    y_out_sample_test_results_optuna_ann=y_out_sample_test_results_optuna_ann_list[i]
    y_out_sample_test_results_optuna_ann=np.concatenate(y_out_sample_test_results_optuna_ann, axis=0 )#convert an array including multiple-arrays to a single array
    

    # Measure the accuracy, f1 score of each fitted models' prediction in the test set within each cv outer split 
    acc_optuna_xgb = accuracy_score(y_test, yhat_optuna_xgb).round(2)*100 
    acc_bayessearach_rnf= accuracy_score(y_test, yhat_bayessearach_rnf).round(2)*100 
    acc_optuna_ann = accuracy_score(y_test, yhat_optuna_ann).round(2)*100 

    f1_out_optuna_xgb=f1_score(y_test, yhat_optuna_xgb).round(2)*100  
    f1_out_bayessearach_rnf=f1_score(y_test, yhat_bayessearach_rnf).round(2)*100  
    f1_out_optuna_ann=f1_score(y_test, yhat_optuna_ann).round(2)*100  

    # Calculate weights of each Classifier based on their own accuracies in each cv outer split
    weight_optuna_xgb=acc_optuna_xgb/(acc_optuna_xgb +acc_bayessearach_rnf+acc_optuna_ann)
    weight_bayessearach_rnf=acc_bayessearach_rnf/(acc_optuna_xgb +acc_bayessearach_rnf+acc_optuna_ann)
    weight_optuna_ann=acc_optuna_ann/(acc_optuna_xgb +acc_bayessearach_rnf+acc_optuna_ann)

    sum_weights=weight_optuna_xgb+weight_bayessearach_rnf+weight_optuna_ann
    print('Sum of weights should be equal to 1. Here is sum_weights=%.3f' % (sum_weights))
    
    #---Obtain ensembled predictions for the test dataset of each cv outer split
    yhat_ensembled=weight_optuna_xgb*yhat_optuna_xgb+weight_bayessearach_rnf*yhat_bayessearach_rnf+weight_optuna_ann*yhat_optuna_ann
    yhat_ensembled=np.where(yhat_ensembled > 0.5, 1,0)
    #---Evaluate the result of optuna xgb, bayes search based RF model, optuna-based ANN (Keras Classifier) and
    #  the  ensembled solution per cv outer split
    acc_optuna_xgb
    acc_bayessearach_rnf
    acc_optuna_ann

    acc_ensembled= accuracy_score(y_test, yhat_ensembled).round(2)*100 
    f1_out_ensembled=f1_score(y_test, yhat_ensembled).round(2)*100  
    # report progress in each cv outer-split
    print('''>out-sample accuracy for each fitted model (xgb, rf, ann, ensembled) acc_xgb=%.3f,
          acc_rnf=%.3f,acc_ann=%.3f, acc_ensem=%.3f''' % (acc_optuna_xgb,acc_bayessearach_rnf,acc_optuna_ann, acc_ensembled) )
    print('''>f1 score for each fitted model (xgb, rf, ann, ensembled) f1_xgb=%.3f,
          f1_rnf=%.3f,  f1_ann=%.3f, f1_ensem=%.3f''' % (f1_out_optuna_xgb,f1_out_bayessearach_rnf,f1_out_optuna_ann,f1_out_ensembled) )
        
    # Public Leaderboard test dataset predictions using each fitted among XGB, RF, KerasClassifier through
    # nested-cross validation models' predictions
    y_out_sample_test_results_optuna_xgb 
    y_out_sample_test_results_bayessearach_rnf    
    y_out_sample_test_results_optuna_ann 

    # Public Leaderboard test dataset predictions using the ensembled Classifier model through
    # nested-cross validation models' predictions    y_out_sample_ensembled=weight_optuna_xgb*y_out_sample_test_results_optuna_xgb+weight_bayessearach_rnf*y_out_sample_test_results_bayessearach_rnf+ weight_optuna_ann*y_out_sample_test_results_optuna_ann
    y_out_sample_ensembled=np.where(y_out_sample_ensembled > 0.5, 1,0)    
    #----- Store the results in each cv outer split
    
    # Optuna-based XGB related lists
    outer_yhat_optuna_xgb.append(yhat_optuna_xgb)
    outer_results_optuna_xgb.append(acc_optuna_xgb)
    outer_results_f1_optuna_xgb.append(f1_out_optuna_xgb)

    # Bayes-Search based Random Forest related lists
    outer_yhat_bayessearach_rnf.append(yhat_bayessearach_rnf)
    outer_results_bayessearach_rnf.append(acc_bayessearach_rnf)
    outer_results_f1_bayessearach_rnf.append(f1_out_bayessearach_rnf)
   
    # Optuna-based ANN (Keras Classifier) related lists
    outer_yhat_optuna_ann.append(yhat_optuna_ann)
    outer_results_optuna_ann.append(acc_optuna_ann)
    outer_results_f1_optuna_ann.append(f1_out_optuna_ann)

    # Ensembled model related lists
    outer_yhat_ensembled.append(yhat_ensembled)
    outer_results_ensembled.append(acc_ensembled)
    outer_results_f1_ensembled.append(f1_out_ensembled)
    #----- Store the ensembled solution predictions for the public leaderboard test dataset in each cv outer split

    y_out_sample_test_results_ensembled.append(y_out_sample_ensembled)
    
    i=i+1



#%% Report the progress of each one fitted model (XGB, RF, Keras) as well as of the ensembled solution across cv outer splits

print('Out-Sample (i.e. per test set within cv outer split) Accuracy: %.3f (%.3f)' % (np.mean(outer_results_optuna_xgb), np.std(outer_results_optuna_xgb))) #Accuracy: 82.600 (2.835)
print('Out-Sample (i.e. per test set within cv outer split) F1 score: %.3f (%.3f)' % (np.mean(outer_results_f1_optuna_xgb), np.std(outer_results_f1_optuna_xgb))) #F1: 76.000 (4.050)

print('Out-Sample (i.e. per test set within cv outer split) Accuracy: %.3f (%.3f)' % (np.mean(outer_results_bayessearach_rnf), np.std(outer_results_bayessearach_rnf))) #Accuracy:82.300 (3.951)
print('Out-Sample (i.e. per test set within cv outer split) F1 score: %.3f (%.3f)' % (np.mean(outer_results_f1_bayessearach_rnf), np.std(outer_results_f1_bayessearach_rnf))) #F1: 75.600 (5.004)

print('Out-Sample (i.e. per test set within cv outer split) Accuracy: %.3f (%.3f)' % (np.mean(outer_results_optuna_ann), np.std(outer_results_optuna_ann))) #Accuracy 80.600 (3.747)
print('Out-Sample (i.e. per test set within cv outer split) F1 score: %.3f (%.3f)' % (np.mean(outer_results_f1_optuna_ann), np.std(outer_results_f1_optuna_ann))) #F1:  73.000 (4.669)

print('Out-Sample (i.e. per test set within cv outer split) Accuracy: %.3f (%.3f)' % (np.mean(outer_results_ensembled), np.std(outer_results_ensembled))) #Accuracy: 81.300 (4.124)
print('Out-Sample (i.e. per test set within cv outer split) F1 score: %.3f (%.3f)' % (np.mean(outer_results_f1_ensembled), np.std(outer_results_f1_ensembled))) #F1:  74.200 (4.976)

#%%%% Save the ensembled solution/model predictions in cv outer splits as well as in the public leaderboard test dataset
directory='C:/Users/vasileios palaskas/Documents/GitHub/Titanic_Classification_Kaggle/models/Ensemble Predictions_XGB_RF_Keras'
logger.info('Define the directory of your saved model objects (folder Titanic_Classification_Kaggle/models/Ensemble Predictions_XGB_RF_Keras')
os.chdir(directory)

# Write-Save objects related to the Nested Cross-validation prodedure implemented through the Ensembled Solution

# CV outer splits: Out-sample results (accuracies)
with open("outer_results_ensembled_xgb_rf_ann", "wb") as outer_results_ensembled_fp:   #Pickling
 pickle.dump(outer_results_ensembled, outer_results_ensembled_fp)

# CV outer splits: Out-sample predictions of the response 
with open("outer_yhat_ensembled_xgb_rf_ann", "wb") as outer_yhat_ensembled_fp:   #Pickling
  pickle.dump(outer_yhat_ensembled, outer_yhat_ensembled_fp)
  
# CV outer splits: Public Leaderboard test dataset predictions of the response   through ensembled solution
with open("y_out_sample_test_results_ensembled_xgb_rf_ann", "wb") as y_out_sample_test_results_ensembled_fp:   #Pickling
 pickle.dump(y_out_sample_test_results_ensembled, y_out_sample_test_results_ensembled_fp)

# Load objects

# CV outer splits: Out-sample results (accuracies)
with open("outer_results_ensembled_xgb_rf_ann", "rb") as outer_results_ensembled_fp:   # Unpickling
   b = pickle.load(outer_results_ensembled_fp)

# CV outer splits: Out-sample predictions of the response 
with open("outer_yhat_ensembled_xgb_rf_ann", "rb") as outer_yhat_ensembled_fp:   # Unpickling
        y_hat_optuna_xgb_list = pickle.load(outer_yhat_ensembled_fp)
        
# CV outer splits: Public Leaderboard test dataset predictions of the response   through ensembled solution
with open("y_out_sample_test_results_ensembled_xgb_rf_ann", "rb") as y_out_sample_test_results_ensembled_fp:   # Unpickling
 b = pickle.load(y_out_sample_test_results_ensembled_fp) 
