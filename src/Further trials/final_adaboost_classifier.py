
#-----------------Ensemble Boosting methods------------------
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Adaboost classifier:Case 1: Use of the Best decision tree classifier by tuning only boosting parameters
#----Adaptive Boosting Classifier--
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(criterion='entropy',
      max_features='log2', min_samples_leaf=1,
   min_samples_split=5,max_depth=5, splitter='best', random_state=0),
    algorithm="SAMME.R", random_state=0) 
ada_clf.fit(X_train_sub_ar, y_train_sub)

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
#---parts of grid search 
#-First 
n_estimators_1= np.array(range(50, 300, 50))
learning_rate_1=np.arange(0.05,0.5,0.05)#The minimum number of samples required to split an internal node:
# define grid search
grid_1 = dict(n_estimators=n_estimators_1,
            learning_rate=learning_rate_1)

#--Combine parameters to a single array (Combine different dictionairies)
ada_parameters =[grid_1]
ada_grid_search = GridSearchCV(estimator = ada_clf,
                           param_grid =ada_parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
ada_grid_search_result = ada_grid_search.fit(X_train_sub_ar, y_train_sub)

ada_best_accuracy = ada_grid_search_result.best_score_
ada_best_parameters =ada_grid_search_result.best_params_
print("Best Accuracy: {:.2f} %".format(ada_best_accuracy*100))# 81.61 %
print("Best Parameters:", ada_best_parameters)# 'learning_rate': 0.05, 'n_estimators': 100
# Adaboost classifier:Case 2 Best decision tree classifier by tuning more detailed boosting parameters



#--Combine parameters to a single array (Combine different dictionairies)
ada_clf_final = AdaBoostClassifier(
    DecisionTreeClassifier(criterion='entropy',
      max_features='log2', min_samples_leaf=1,
   min_samples_split=5,max_depth=5, splitter='best', random_state=0),n_estimators=100,
    learning_rate=0.05,
    algorithm="SAMME.R", random_state=0)
    

ada_clf_final_result=ada_clf_final.fit(X_train_sub_ar, y_train_sub)

# Predicting the Test set results
y_pred_1 = ada_clf_final_result.predict(X_test_sub_ar)
y_train_pred_1 = ada_clf_final_result.predict(X_train_sub_ar)


# Making the Confusion Matrix and the accuracy score
cm_1 = confusion_matrix(y_test_sub, y_pred_1)

print(cm_1)  # =

accuracy_score(y_test_sub, y_pred_1).round(2)*100  # 82.0 %%
accuracy_score(y_train_sub, y_train_pred_1).round(2)*100  # 97.0%
f1_score(y_test_sub, y_pred_1).round(2)*100  # 74.0%%



#-More detailed search for the parameters in order to improve the accuracy
n_estimators_2= np.array(range(50, 200, 10))
learning_rate_2=[0.05]#The minimum number of samples required to split an internal node:
# define grid search
grid_2 = dict(n_estimators=n_estimators_2,
            learning_rate=learning_rate_2)

#--Combine parameters to a single array (Combine different dictionairies)
ada_parameters =[grid_2]
ada_grid_search = GridSearchCV(estimator = ada_clf,
                           param_grid =ada_parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
ada_grid_search_result = ada_grid_search.fit(X_train_sub_ar, y_train_sub)

ada_best_accuracy = ada_grid_search_result.best_score_
ada_best_parameters =ada_grid_search_result.best_params_
print("Best Accuracy: {:.2f} %".format(ada_best_accuracy*100))#  81.75
print("Best Parameters:", ada_best_parameters)# 'learning_rate': 0.05, 'n_estimators': 110

#---Parameters after a second more detailed grid search performs the best.

#--Combine parameters to a single array (Combine different dictionairies)
ada_clf_final = AdaBoostClassifier(
    DecisionTreeClassifier(criterion='entropy',
      max_features='log2', min_samples_leaf=1,
   min_samples_split=5,max_depth=5, splitter='best', random_state=0),n_estimators=110,
    learning_rate=0.05,
    algorithm="SAMME.R", random_state=0)
    

ada_clf_final_result=ada_clf_final.fit(X_train_sub_ar, y_train_sub)

# Predicting the Test set results
y_pred_1 = ada_clf_final_result.predict(X_test_sub_ar)
y_train_pred_1 = ada_clf_final_result.predict(X_train_sub_ar)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Making the Confusion Matrix and the accuracy score
cm_1 = confusion_matrix(y_test_sub, y_pred_1)

print(cm_1)  # =
#--Final in-sample and out-sample results using accuracy and f1 scores due to imbalanced classes
accuracy_score(y_test_sub, y_pred_1).round(2)*100  # 81.0 %%
accuracy_score(y_train_sub, y_train_pred_1).round(2)*100  # 97.0%
f1_score(y_test_sub, y_pred_1).round(2)*100  # 73.0%%


