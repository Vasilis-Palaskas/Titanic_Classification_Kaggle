#--------Gradient Boosting Classifier-Method 1: Use best decision classifier and tune only boosting parameters

#----Tune gradient boosting classifer learning rate and  n_estimators. However, instead of guessing randomly 
#---- initial values  for tree parameters, we will use as starting point the best ones from a  single decision tree classifier
from sklearn.ensemble import GradientBoostingClassifier
gbrt=GradientBoostingClassifier(max_features='log2', min_samples_leaf=1,
   min_samples_split=5,max_depth=5,random_state=0)
gbrt.fit(X_train_sub_ar, y_train_sub)


#---Round 1: Tune the learning rate
from sklearn.model_selection import GridSearchCV
learning_rate_1=np.arange(0.01,0.2,0.01)#The minimum number of samples required to split an internal node:

grid_1 = dict(
            learning_rate=learning_rate_1  )

#--Combine parameters to a single array (Combine different dictionairies)
gbrt_parameters =[grid_1]
gbrt_grid_search = GridSearchCV(estimator = gbrt,
                           param_grid =gbrt_parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
gbrt_grid_search_result =gbrt_grid_search.fit(X_train_sub_ar, y_train_sub)


gbrt_best_accuracy = gbrt_grid_search_result.best_score_
gbrt_best_parameters =gbrt_grid_search_result.best_params_
print("Best Accuracy: {:.2f} %".format(gbrt_best_accuracy*100))# 83.43  %
print("Best Parameters:", gbrt_best_parameters)# 'learning_rate': 0.04,

#---Round 2: Tune the n_estimators based on the learning rate

n_estimators_2=np.array(range(10, 300, 20))

grid_2 = dict(   n_estimators=n_estimators_2  )
gbrt_2=GradientBoostingClassifier(max_features='log2', min_samples_leaf=1,
   min_samples_split=5,max_depth=5,random_state=0,learning_rate=0.04)
gbrt_2.fit(X_train_sub_ar, y_train_sub)
#--Combine parameters to a single array (Combine different dictionairies)
gbrt_parameters =[grid_2]
gbrt_grid_search = GridSearchCV(estimator = gbrt_2,
                           param_grid =gbrt_parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
gbrt_grid_search_result =gbrt_grid_search.fit(X_train_sub_ar, y_train_sub)


gbrt_best_accuracy = gbrt_grid_search_result.best_score_
gbrt_best_parameters =gbrt_grid_search_result.best_params_
print("Best Accuracy: {:.2f} %".format(gbrt_best_accuracy*100))# 83.71  %
print("Best Parameters:", gbrt_best_parameters)# 'learning_rate': 0.04, 'n_estimators': 90

#---Until now the solution is identical with gradient_boosting_1 grid search. This means that for the
#-- initial tuning of boosting parameters, we need to adopt the parameters of a best decision tree classifier
#-- and then, implement a grid search for both learning rate and n estimator parameters.
#--Now we will proceed to make a tuning in tree parameters and then a re-grid search for boosting parameters

#---Round 3: Tune the most important tree parameters based on specified learning rate and n_estimators

min_samples_split_3=np.array(range(2, 80, 3))
max_depth_3=np.array(range(2, 20, 1))
# define grid search
grid_3 = dict(min_samples_split=min_samples_split_3,
             max_depth= max_depth_3)
gbrt_3=GradientBoostingClassifier(n_estimators=90,
                                  random_state=0,learning_rate=0.04)
gbrt_3.fit(X_train_sub_ar, y_train_sub)
#--Combine parameters to a single array (Combine different dictionairies)
gbrt_parameters =[grid_3]
gbrt_grid_search = GridSearchCV(estimator = gbrt_3,
                           param_grid =gbrt_parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
gbrt_grid_search_result =gbrt_grid_search.fit(X_train_sub_ar, y_train_sub)


gbrt_best_accuracy = gbrt_grid_search_result.best_score_
gbrt_best_parameters =gbrt_grid_search_result.best_params_
print("Best Accuracy: {:.2f} %".format(gbrt_best_accuracy*100))# 83.71 %
print("Best Parameters:", gbrt_best_parameters)#'max_depth': 14, 'min_samples_split': 65}

#--Combine parameters to a single array (Combine different dictionairies)
gbrt_4=GradientBoostingClassifier(n_estimators=90,
                                  random_state=0,learning_rate=0.04,max_depth=14)
    

gbrt_4_result=gbrt_4.fit(X_train_sub_ar, y_train_sub)

# Predicting the Test set results
y_pred_1 = gbrt_4.predict(X_test_sub_ar)
y_train_pred_1 = gbrt_4_result.predict(X_train_sub_ar)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Making the Confusion Matrix and the accuracy score
cm_1 = confusion_matrix(y_test_sub, y_pred_1)

print(cm_1)  # =

accuracy_score(y_test_sub, y_pred_1).round(2)*100  # 81.0 %%
accuracy_score(y_train_sub, y_train_pred_1).round(2)*100  # 98.0%

f1_score(y_test_sub, y_pred_1).round(2)*100  # 74.0%%
#---Round 4: Tune the secondary most important tree parameters based on the remaining pre-specified parameters

min_samples_split_4=np.array(range(60,70, 1))
min_samples_leaf_4=np.array(range(1, 15, 1))
# define grid search
grid_4 = dict(min_samples_split=min_samples_split_4,
             min_samples_leaf=min_samples_leaf_4)
gbrt_4=GradientBoostingClassifier(n_estimators=90,
                                  random_state=0,learning_rate=0.04,max_depth=14)
gbrt_4.fit(X_train_sub_ar, y_train_sub)
#--Combine parameters to a single array (Combine different dictionairies)
gbrt_parameters =[grid_4]
gbrt_grid_search = GridSearchCV(estimator = gbrt_4,
                           param_grid =gbrt_parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
gbrt_grid_search_result =gbrt_grid_search.fit(X_train_sub_ar, y_train_sub)


gbrt_best_accuracy = gbrt_grid_search_result.best_score_
gbrt_best_parameters =gbrt_grid_search_result.best_params_
print("Best Accuracy: {:.2f} %".format(gbrt_best_accuracy*100))# 83.71  %
print("Best Parameters:", gbrt_best_parameters)# Best Parameters: {'min_samples_leaf': 1, 'min_samples_split': 65}




#---Round 5: Tune the secondary most important tree parameters based on the remaining pre-specified parameters
max_features_5=['log2',"sqrt","auto"] 

# define grid search
grid_5 = dict(max_features=max_features_5)
gbrt_5=GradientBoostingClassifier(n_estimators=90,
                                  random_state=0,learning_rate=0.04,max_depth=14,
                                  min_samples_split=65)
gbrt_5.fit(X_train_sub_ar, y_train_sub)
#--Combine parameters to a single array (Combine different dictionairies)
gbrt_parameters =[grid_5]
gbrt_grid_search = GridSearchCV(estimator = gbrt_5,
                           param_grid =gbrt_parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
gbrt_grid_search_result =gbrt_grid_search.fit(X_train_sub_ar, y_train_sub)


gbrt_best_accuracy = gbrt_grid_search_result.best_score_
gbrt_best_parameters =gbrt_grid_search_result.best_params_
print("Best Accuracy: {:.2f} %".format(gbrt_best_accuracy*100))# 83.29  %
print("Best Parameters:", gbrt_best_parameters)# Best Parameters: {'min_samples_leaf': 1, 'min_samples_split': 33, {'max_features': 'auto'}}

#---Round 6: Tune subsample in order to check whether a stohastic gradient boosting is required or not
subsample_6=[0.5,0.6,0.7,0.8,0.9,1] 

# define grid search
grid_6 = dict(subsample=subsample_6)
gbrt_6=GradientBoostingClassifier(n_estimators=90,
                                  random_state=0,learning_rate=0.04,max_depth=14,
                                  min_samples_split=65)
gbrt_6.fit(X_train_sub_ar, y_train_sub)
#--Combine parameters to a single array (Combine different dictionairies)
gbrt_parameters =[grid_6]
gbrt_grid_search = GridSearchCV(estimator = gbrt_6,
                           param_grid =gbrt_parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
gbrt_grid_search_result =gbrt_grid_search.fit(X_train_sub_ar, y_train_sub)


gbrt_best_accuracy = gbrt_grid_search_result.best_score_
gbrt_best_parameters =gbrt_grid_search_result.best_params_
print("Best Accuracy: {:.2f} %".format(gbrt_best_accuracy*100))#  83.71%
print("Best Parameters:", gbrt_best_parameters)# Best Parameters: {'subsample': 0.9}

#---Round 7: Re-Tune learning rate and n estimator parameters proportionally by the opposite direction between each other
learning_rate_7=np.arange(0.01,0.1,0.01)#The minimum number of samples required to split an internal node:
n_estimators_7=np.arange(40,100,5)#The minimum number of samples required to split an internal node:

# define grid search
grid_7 = dict(learning_rate=learning_rate_7,n_estimators=n_estimators_7)
gbrt_7=GradientBoostingClassifier(
                                  random_state=0,max_depth=14,
                                  min_samples_split=65)
gbrt_7.fit(X_train_sub_ar, y_train_sub)
#--Combine parameters to a single array (Combine different dictionairies)
gbrt_parameters =[grid_7]
gbrt_grid_search = GridSearchCV(estimator = gbrt_7,
                           param_grid =gbrt_parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
gbrt_grid_search_result =gbrt_grid_search.fit(X_train_sub_ar, y_train_sub)


gbrt_best_accuracy = gbrt_grid_search_result.best_score_
gbrt_best_parameters =gbrt_grid_search_result.best_params_
print("Best Accuracy: {:.2f} %".format(gbrt_best_accuracy*100))#  83.71%
print("Best Parameters:", gbrt_best_parameters)# Best Parameters: {'learning_rate': 0.04, 'n_estimators': 90}

#-----Round 8: Tune some additional hyperparameters
loss_8=['deviance', 'exponential']
criterion_8=["friedman_mse", "squared_error", "mse", "mae","friedman_mse"]

# define grid search
grid_8 = dict(loss=loss_8,criterion=criterion_8)
gbrt_8=GradientBoostingClassifier(n_estimators=90,
                                  random_state=0,learning_rate=0.04,max_depth=14,
                                  min_samples_split=65)
gbrt_8.fit(X_train_sub_ar, y_train_sub)
#--Combine parameters to a single array (Combine different dictionairies)
gbrt_parameters =[grid_8]
gbrt_grid_search = GridSearchCV(estimator = gbrt_8,
                           param_grid =gbrt_parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
gbrt_grid_search_result =gbrt_grid_search.fit(X_train_sub_ar, y_train_sub)


gbrt_best_accuracy = gbrt_grid_search_result.best_score_
gbrt_best_parameters =gbrt_grid_search_result.best_params_
print("Best Accuracy: {:.2f} %".format(gbrt_best_accuracy*100))#  83.71%
print("Best Parameters:", gbrt_best_parameters)# Best Parameters: {'learning_rate': 0.04, 'n_estimators': 90}


#----Final Gradient boosting classifier after a long grid search
gbrt_clf_final =GradientBoostingClassifier(n_estimators=90,
                                  random_state=0,learning_rate=0.04,max_depth=14,
                                  min_samples_split=65)
    

gbrt_clf_final_result=gbrt_clf_final.fit(X_train_sub_ar, y_train_sub)

# Predicting the Test set results
y_pred_1 = gbrt_clf_final.predict(X_test_sub_ar)
y_train_pred_1 = gbrt_clf_final_result.predict(X_train_sub_ar)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Making the Confusion Matrix and the accuracy score
cm_1 = confusion_matrix(y_test_sub, y_pred_1)

print(cm_1)  # =

accuracy_score(y_test_sub, y_pred_1).round(2)*100  # 86.0 %%
accuracy_score(y_train_sub, y_train_pred_1).round(2)*100  # 92.0%

precision_score(y_test_sub, y_pred_1).round(2)*100  #87.0 %%
recall_score(y_test_sub, y_pred_1).round(2)*100  # 75.0 %%
f1_score(y_test_sub, y_pred_1).round(2)*100  # 81.0%%
