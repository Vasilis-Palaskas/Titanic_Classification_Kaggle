
#%%% Nested cross-validation predictions ensembling
#---In essence, for each cv outer split (10 in total) we fit the best model
#-- (obtained from hyperparams search implemented in 6 cv inner splits) and we obtain the predictions
#-- of the public leaderboard test set using the fitted model in each one from 10 cv outer splits. This
#-- process is implemented for both XGBoost and Random Forest classifier and then we ensemble their predictions
#-- in the public leaderboard test dataset using weighted averaging. It is essential to mention
#-- that in order to decide whether the ensembling provide better predictions than the training of either
#-- classifiers (either XGB or RF) we evaluated the ensembling technique by evaluating the predictions
#-- of models trains in each cv outer split.


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
#-- Ensembled models related lists to be initialised for each cv outer split
outer_yhat_ensembled=list()
outer_results_ensembled = list()
outer_results_f1_ensembled = list()
y_out_sample_test_results_ensembled=list()


i=0# counting the cv outer splits
# Now the nested cross validation begins
for train_ix, test_ix in cv_outer.split(X,y):
    print(i+1)

    #---- split data
    X_train = X.iloc[train_ix,:]
    X_test = X.iloc[test_ix,:]
    y_train = y.iloc[train_ix]
    y_test = y.iloc[test_ix]
    
    # The corresponding predictions for the test dataset of each cv outer split from each fitted model  in each cv outer split

    yhat_optuna_xgb=yhat_optuna_xgb_list[i]# remove the last column in terms of removing NAN
    yhat_bayessearach_rnf=yhat_bayessearach_rnf_list[i]# remove the last column in terms of removing NAN
    
    # The corresponding predictions for the public leaderboard test dataset from each fitted model in each cv outer split
    y_out_sample_test_results_optuna_xgb=y_out_sample_test_results_optuna_xgb_list[i]
    y_out_sample_test_results_bayessearach_rnf=y_out_sample_test_results_bayessearach_rnf_list[i] 
    
    
    # Measure the accuracy, f1 score of each fitted models' prediction in the test set within each cv outer split 
    acc_optuna_xgb = accuracy_score(y_test, yhat_optuna_xgb).round(2)*100 
    acc_bayessearach_rnf= accuracy_score(y_test, yhat_bayessearach_rnf).round(2)*100 

    f1_out_optuna_xgb=f1_score(y_test, yhat_optuna_xgb).round(2)*100  
    f1_out_bayessearach_rnf=f1_score(y_test, yhat_bayessearach_rnf).round(2)*100  

    # Calculate weights of each Classifier based on their own accuracies in each cv outer split
    weight_optuna_xgb=acc_optuna_xgb/(acc_optuna_xgb +acc_bayessearach_rnf)
    weight_bayessearach_rnf=acc_bayessearach_rnf/(acc_optuna_xgb +acc_bayessearach_rnf)
    sum_weights=weight_optuna_xgb+weight_bayessearach_rnf
    print('Sum of weights should be equal to 1. Here is sum_weights=%.3f' % (sum_weights))
    
    #---Obtain ensembled predictions for the test dataset of each cv outer split
    yhat_ensembled=weight_optuna_xgb*yhat_optuna_xgb+weight_bayessearach_rnf*yhat_bayessearach_rnf
    yhat_ensembled=np.where(yhat_ensembled > 0.5, 1,0)
    #---Evaluate the result of optuna xgb, bayes search based RF model and
    #   the ensembled solution per cv outer split
    acc_optuna_xgb
    acc_bayessearach_rnf
    acc_ensembled= accuracy_score(y_test, yhat_ensembled).round(2)*100 
    f1_out_ensembled=f1_score(y_test, yhat_ensembled).round(2)*100  
    # report progress in each cv outer-split
    print('''>out-sample accuracy for each fitted model (xgb, rf and ensembled) acc_xgb=%.3f,
          acc_rnf=%.3f, acc_ensem=%.3f''' % (acc_optuna_xgb,acc_bayessearach_rnf, acc_ensembled) )
    print('''>f1 score for each fitted model (xgb, rf and ensembled) f1_xgb=%.3f,
          f1_rnf=%.3f, f1_ensem=%.3f''' % (f1_out_optuna_xgb,f1_out_bayessearach_rnf,f1_out_ensembled) )
        
    # Public Leaderboard test dataset predictions using each fitted among XGB, RF through
    # nested-cross validation models' predictions
    y_out_sample_test_results_optuna_xgb 
    y_out_sample_test_results_bayessearach_rnf    

    # Public Leaderboard test dataset predictions using the ensembled Classifier model through
    # nested-cross validation models' predictions  
    y_out_sample_ensembled=weight_optuna_xgb*y_out_sample_test_results_optuna_xgb+weight_bayessearach_rnf*y_out_sample_test_results_bayessearach_rnf
    y_out_sample_ensembled=weight_optuna_xgb*y_out_sample_test_results_optuna_xgb+weight_bayessearach_rnf*y_out_sample_test_results_bayessearach_rnf
    y_out_sample_ensembled=np.where(y_out_sample_ensembled > 0.5, 1,0)    
    
    #----- Store the results in each cv outer split
    # Optuna-based XGB related lists    outer_yhat_optuna_xgb.append(yhat_optuna_xgb)
    outer_results_optuna_xgb.append(acc_optuna_xgb)
    outer_results_f1_optuna_xgb.append(f1_out_optuna_xgb)

    #  Bayesearch-based Random Forest related lists
    outer_yhat_bayessearach_rnf.append(yhat_bayessearach_rnf)
    outer_results_bayessearach_rnf.append(acc_bayessearach_rnf)
    outer_results_f1_bayessearach_rnf.append(f1_out_bayessearach_rnf)

    # Ensembled
    outer_yhat_ensembled.append(yhat_ensembled)
    outer_results_ensembled.append(acc_ensembled)
    outer_results_f1_ensembled.append(f1_out_ensembled)
    
    #----- Store the ensembled solution predictions for the public leaderboard test dataset in each cv outer split
    y_out_sample_test_results_ensembled.append(y_out_sample_ensembled)
    
    i=i+1



#%% Report the progress of each one fitted model (XGB, RF) as well as of the ensembled solution across cv outer splits

print('Out-Sample (i.e. per test set within cv outer split) Accuracy: %.3f (%.3f)' % (np.mean(outer_results_optuna_xgb), np.std(outer_results_optuna_xgb))) #Accuracy:82.600 (2.835)
print('Out-Sample (i.e. per test set within cv outer split) F1 score: %.3f (%.3f)' % (np.mean(outer_results_f1_optuna_xgb), np.std(outer_results_f1_optuna_xgb))) #F1:  76.000 (4.050)

print('Out-Sample (i.e. per test set within cv outer split) Accuracy: %.3f (%.3f)' % (np.mean(outer_results_bayessearach_rnf), np.std(outer_results_bayessearach_rnf))) #Accuracy: 82.300 (3.951)
print('Out-Sample (i.e. per test set within cv outer split) F1 score: %.3f (%.3f)' % (np.mean(outer_results_f1_bayessearach_rnf), np.std(outer_results_f1_bayessearach_rnf))) #F1:   75.600 (5.004)

print('Out-Sample (i.e. per test set within cv outer split) Accuracy: %.3f (%.3f)' % (np.mean(outer_results_ensembled), np.std(outer_results_ensembled))) #Accuracy:    83.500 (3.263)
print('Out-Sample (i.e. per test set within cv outer split) F1 score: %.3f (%.3f)' % (np.mean(outer_results_f1_ensembled), np.std(outer_results_f1_ensembled))) #F1:  77.200 (4.895)

#%%%% Save the ensembled solution/model predictions in cv outer splits as well as in the public leaderboard test dataset

directory='C:/Users/vasileios palaskas/Documents/GitHub/Titanic_Classification_Kaggle/models/Ensemble Predictions_XGB_RF'
logger.info('Define the directory of your saved model objects (folder Titanic_Classification_Kaggle/models/Ensemble Predictions_XGB_RF')
os.chdir(directory)

# Write-Save objects related to the Nested Cross-validation prodedure implemented through the Ensembled Solution

# CV outer splits: Out-sample results (accuracies)
with open("outer_results_ensembled", "wb") as outer_results_ensembled_fp:   #Pickling
 pickle.dump(outer_results_ensembled, outer_results_ensembled_fp)

# CV outer splits: Out-sample predictions of the response 
with open("y_out_sample_test_results_ensembled", "wb") as y_out_sample_test_results_ensembled_fp:   #Pickling
 pickle.dump(y_out_sample_test_results_ensembled, y_out_sample_test_results_ensembled_fp)
 
# CV outer splits: Public Leaderboard test dataset predictions of the response   through ensembled solution
 with open("outer_yhat_ensembled", "wb") as outer_yhat_ensembled_fp:   #Pickling
  pickle.dump(outer_yhat_ensembled, outer_yhat_ensembled_fp)

# Load objects

# CV outer splits: Out-sample results (accuracies)
with open("outer_results_ensembled", "rb") as outer_results_ensembled_fp:   # Unpickling
 b = pickle.load(outer_results_ensembled_fp)

# CV outer splits: Out-sample predictions of the response 
with open("outer_yhat_ensembled", "rb") as outer_yhat_ensembled_fp:   # Unpickling
        y_hat_optuna_xgb_list = pickle.load(outer_yhat_ensembled_fp)

# CV outer splits: Public Leaderboard test dataset predictions of the response   through ensembled solution
 with open("outer_yhat_ensembled", "rb") as outer_yhat_ensembled_fp:   #Pickling
  pickle.dump(outer_yhat_ensembled, outer_yhat_ensembled_fp)

