#% Load saved objects from Python scripts: Optuna_Search_XGB.py and Bayes_Search_RF.py
 # to use them as ingredients to be ensembled



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
    
    #-----Transformations  of predictions per cv outer-split

    # Keep for each cv outer split the corresponding predictions from each fitted model
    yhat_optuna_xgb=yhat_optuna_xgb_list[i]# remove the last column in terms of removing NAN
    yhat_bayessearach_rnf=yhat_bayessearach_rnf_list[i]# remove the last column in terms of removing NAN
    
    # Keep for each cv outer split the corresponding predictions from each fitted model
    y_out_sample_test_results_optuna_xgb=y_out_sample_test_results_optuna_xgb_list[i]
    y_out_sample_test_results_bayessearach_rnf=y_out_sample_test_results_bayessearach_rnf_list[i] 
    
    
    # Measure the accuracy, f1 score of each fitted models' prediction in the test set within each cv outer split 
    acc_optuna_xgb = accuracy_score(y_test, yhat_optuna_xgb).round(2)*100 
    acc_bayessearach_rnf= accuracy_score(y_test, yhat_bayessearach_rnf).round(2)*100 

    f1_out_optuna_xgb=f1_score(y_test, yhat_optuna_xgb).round(2)*100  
    f1_out_bayessearach_rnf=f1_score(y_test, yhat_bayessearach_rnf).round(2)*100  

    # Using averaging weights
    weight_optuna_xgb=acc_optuna_xgb/(acc_optuna_xgb +acc_bayessearach_rnf)
    weight_bayessearach_rnf=acc_bayessearach_rnf/(acc_optuna_xgb +acc_bayessearach_rnf)
    sum_weights=weight_optuna_xgb+weight_bayessearach_rnf
    print('Sum of weights should be equal to 1. Here is sum_weights=%.3f' % (sum_weights))
    
    #---Obtain ensembled predictions
    yhat_ensembled=weight_optuna_xgb*yhat_optuna_xgb+weight_bayessearach_rnf*yhat_bayessearach_rnf
    yhat_ensembled=np.where(yhat_ensembled > 0.5, 1,0)
    #---Evaluare the result of optuna xgb, bayes search based RF model and ensembled
    #   per cv outer split
    acc_optuna_xgb
    acc_bayessearach_rnf
    acc_ensembled= accuracy_score(y_test, yhat_ensembled).round(2)*100 
    f1_out_ensembled=f1_score(y_test, yhat_ensembled).round(2)*100  
    print('''>out-sample accuracy for each fitted model (xgb, rf and ensembled) acc_xgb=%.3f,
          acc_rnf=%.3f, acc_ensem=%.3f''' % (acc_optuna_xgb,acc_bayessearach_rnf, acc_ensembled) )
    print('''>f1 score for each fitted model (xgb, rf and ensembled) f1_xgb=%.3f,
          f1_rnf=%.3f, f1_ensem=%.3f''' % (f1_out_optuna_xgb,f1_out_bayessearach_rnf,f1_out_ensembled) )
        
    #-------Competition Test set predictions using nested-cross val runs per fitted model
    y_out_sample_test_results_optuna_xgb 
    y_out_sample_test_results_bayessearach_rnf    

    # calculate ensembled
    y_out_sample_ensembled=weight_optuna_xgb*y_out_sample_test_results_optuna_xgb+weight_bayessearach_rnf*y_out_sample_test_results_bayessearach_rnf
    y_out_sample_ensembled=np.where(y_out_sample_ensembled > 0.5, 1,0)    
    #----- Store the results
    # XGB
    outer_yhat_optuna_xgb.append(yhat_optuna_xgb)
    outer_results_optuna_xgb.append(acc_optuna_xgb)
    outer_results_f1_optuna_xgb.append(f1_out_optuna_xgb)
    #!!! y_out_sample_test_results_optuna_xgb.append(y_out_sample_test_results_optuna_xgb)

    #  Bayesearch Random Forest related lists
    outer_yhat_bayessearach_rnf.append(yhat_bayessearach_rnf)
    outer_results_bayessearach_rnf.append(acc_bayessearach_rnf)
    outer_results_f1_bayessearach_rnf.append(f1_out_bayessearach_rnf)
    #!!!y_out_sample_test_results_bayessearach_rnf.append(y_out_sample_test_results_bayessearach_rnf)

    # Ensembled
    outer_yhat_ensembled.append(yhat_ensembled)
    outer_results_ensembled.append(acc_ensembled)
    outer_results_f1_ensembled.append(f1_out_ensembled)
    y_out_sample_test_results_ensembled.append(y_out_sample_ensembled)
    
    i=i+1



#%% Output of Optuna search and predictions

#--- summarize the estimated performance of the model
print('Out-Sample (i.e. per test set within cv outer split) Accuracy: %.3f (%.3f)' % (np.mean(outer_results_optuna_xgb), np.std(outer_results_optuna_xgb))) #Accuracy: 83.600 (4.224)
print('Out-Sample (i.e. per test set within cv outer split) F1 score: %.3f (%.3f)' % (np.mean(outer_results_f1_optuna_xgb), np.std(outer_results_f1_optuna_xgb))) #F1:  77.100 (5.576)

print('Out-Sample (i.e. per test set within cv outer split) Accuracy: %.3f (%.3f)' % (np.mean(outer_results_bayessearach_rnf), np.std(outer_results_bayessearach_rnf))) #Accuracy: 80.800 (1.990)
print('Out-Sample (i.e. per test set within cv outer split) F1 score: %.3f (%.3f)' % (np.mean(outer_results_f1_bayessearach_rnf), np.std(outer_results_f1_bayessearach_rnf))) #F1:  73.700 (2.968)

print('Out-Sample (i.e. per test set within cv outer split) Accuracy: %.3f (%.3f)' % (np.mean(outer_results_ensembled), np.std(outer_results_ensembled))) #Accuracy:   83.800 (3.789)
print('Out-Sample (i.e. per test set within cv outer split) F1 score: %.3f (%.3f)' % (np.mean(outer_results_f1_ensembled), np.std(outer_results_f1_ensembled))) #F1:   77.100 (5.356)

#%%%% Store our files
with open("outer_results_ensembled", "wb") as outer_results_ensembled_fp:   #Pickling
 pickle.dump(outer_results_ensembled, outer_results_ensembled_fp)

with open("outer_results_ensembled", "rb") as outer_results_ensembled_fp:   # Unpickling
 b = pickle.load(outer_results_ensembled_fp)
#
with open("y_out_sample_test_results_ensembled", "wb") as y_out_sample_test_results_ensembled_fp:   #Pickling
 pickle.dump(y_out_sample_test_results_ensembled, y_out_sample_test_results_ensembled_fp)

with open("y_out_sample_test_results_ensembled", "rb") as y_out_sample_test_results_ensembled_fp:   # Unpickling
 b = pickle.load(y_out_sample_test_results_ensembled_fp) 
#

 #
 with open("outer_yhat_ensembled", "wb") as outer_yhat_ensembled_fp:   #Pickling
  pickle.dump(outer_yhat_ensembled, outer_yhat_ensembled_fp)
with open("outer_yhat_ensembled", "rb") as outer_yhat_ensembled_fp:   # Unpickling
        y_hat_optuna_xgb_list = pickle.load(outer_yhat_ensembled_fp)
