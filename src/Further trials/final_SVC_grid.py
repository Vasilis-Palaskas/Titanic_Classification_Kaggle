#-----Support Vector Classification (SVC)-----

#--SVC General classifier
from sklearn.svm import SVC
svc_classifier = SVC()
svc_classifier.fit(X_train_sub_ar, y_train_sub)

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
#---parts of grid search because each solver depends on the penalty chosen
#-First 
#loss_1 = ['hinge', 'squared_hinge']
#penalty_1 = ['l2']
kernel_1=['rbf']
c_values_1 = [np.array(range(0,100,5)), 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2,
             0.1, 0.01]
gamma_values_1= [ 0.05,0.1,0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95 ]
#-second
#loss_2 = [ 'squared_hinge']
#penalty_2 = ['l1']
kernel_2=['rbf']
c_values_2 = [np.array(range(0,100,5)), 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2,
             0.1, 0.01]
gamma_values_2=[ 0.05,0.1,0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95 ]

#-third
#loss_3 = ['hinge', 'squared_hinge']
#penalty_3 = ['l2']
kernel_3=['linear']
c_values_3 = [np.array(range(0,100,5)), 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2,
             0.1, 0.01]
#-fourth
#loss_4 = [ 'squared_hinge']
#penalty_4 = ['l1']
kernel_4=['linear']
c_values_4 =  [np.array(range(0,100,5)), 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2,
             0.1, 0.01]
#-fifth
kernel_5=['poly']
degree_5=[2,3]
coef0_5=[1,1.5,2]
c_values_5 =  [np.array(range(0,100,5)), 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2,
             0.1, 0.01]
#-sixth
kernel_6=['sigmoid']
coef0_6=np.array(range(0,100,1))
c_values_6 =  [np.array(range(0,100,5)), 0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2,
             0.1, 0.01]
# define grid search
grid_1 = dict(kernel=kernel_1,
              C=c_values_1,  gamma=gamma_values_1)
grid_2 = dict(kernel=kernel_2,
              C=c_values_2, gamma=gamma_values_2)
grid_3 = dict(kernel=kernel_3,
             C=c_values_3)
grid_4 = dict(kernel=kernel_4,
              C=c_values_4)
grid_5 = dict(kernel=kernel_5,degree=degree_5,coef0=coef0_5,
              C=c_values_5)
grid_6 = dict(kernel=kernel_6,coef0=coef0_6,
              C=c_values_6)
#--Combine parameters to a single array (Combine different dictionairies)
svc_parameters =[grid_1,grid_2,grid_3 ,grid_4,grid_5,
              grid_6]
svc_grid_search = GridSearchCV(estimator = svc_classifier,
                           param_grid = svc_parameters ,
                           scoring = 'accuracy',random_state=0,
                           cv = 10,
                           n_jobs = -1)
svc_grid_search_result = svc_grid_search.fit(X_train_sub_ar, y_train_sub)
best_accuracy = svc_grid_search_result.best_score_
best_parameters = svc_grid_search_result.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))#83.58 %
print("Best Parameters:", best_parameters)#Best Parameters: {'C': 2, 'gamma': 0.1, 'kernel': 'rbf'}
#{'C': 1.4, 'gamma': 0.15, 'kernel': 'rbf'}

# Predicting the Out of sample as well as In sample predictive results based on final logistic classifier

svc_classifier = SVC(C=1.4, gamma=0.15,kernel="rbf",random_state=0)
svc_classifier.fit(X_train_sub_ar, y_train_sub)

y_pred_1 = svc_classifier.predict(X_test_sub_ar)
y_train_pred_1 = svc_classifier.predict(X_train_sub_ar)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Making the Confusion Matrix and the accuracy score
cm_1 = confusion_matrix(y_test_sub, y_pred_1)

print(cm_1)
#--Accuracy score in both test and train sets as well as the f1 score in test sets
accuracy_score(y_test_sub, y_pred_1).round(2)*100  # 80.0 %%
accuracy_score(y_train_sub, y_train_pred_1).round(2)*100  # 84.0%

f1_score(y_test_sub, y_pred_1).round(2)*100  # 72.0%%
#---more detailed grid search based on the results of previous grid searches
#-First 
kernel_5=['rbf']
c_values_5 = np.arange(1.3, 3.9, 0.1)
gamma_values_5=[ 0.05,0.1,0.15, 0.2 ]
#grid_5 = dict(kernel=kernel_5,
#              C=c_values_5,  gamma=gamma_values_5)
grid_5 = dict(kernel=kernel_5,
              C=c_values_5, gamma=gamma_values_5)
svc_grid_search_5 = GridSearchCV(estimator = svc_classifier,
                           param_grid = grid_5,
                           scoring = 'accuracy',
                           cv = 10,random_state=0,
                           n_jobs = -1)
svc_grid_search_5_result = svc_grid_search_5.fit(X_train_sub_ar, y_train_sub)
dir(svc_grid_search_5_result)

#--Print the best results in accuracy along with the corresponding parameter
print("Best: %2f using %2s" % (svc_grid_search_5_result .best_score_*100,
           svc_grid_search_5_result .best_params_))#83.72 using {'C': 2.2, 'gamma': 0.1, 'kernel': 'rbf'}

best_means_5 = svc_grid_search_5_result.cv_results_['mean_test_score'].max()
best_means_5_index=np.where(svc_grid_search_5_result.cv_results_['mean_test_score']==best_means_5 )
best_stds_5=svc_grid_search_5_result.cv_results_['std_test_score'][best_means_5_index][0]
#Print accuracy with sd
print("Best Accuracy: {:.2f} %".format(best_means_5*100))#Best Accuracy: 83.72 %
print("Std of Accuracy: {:.2f} %".format(best_stds_5*100))#Std of Accuracy: 3.09 %

#---All means, stds and their parameters and their display
means_5 = svc_grid_search_5_result.cv_results_['mean_test_score']
stds_5 = svc_grid_search_5_result.cv_results_['std_test_score']
params_5 = svc_grid_search_5_result.cv_results_['params']
for mean, stdev, param in zip(means_5, stds_5, params_5):
    print("%f (%f) with: %r" % (mean, stdev, param))

#-----Final SVC results-----
#-Fitting SVC using best params
best_svc_classifier_ridge =SVC(C=2.2,gamma=0.1,
                               kernel="rbf",
                               random_state = 0)

best_svc_classifier_ridge.fit(X_train_sub_ar, y_train_sub)

dir(best_svc_classifier_ridge)
best_svc_classifier_ridge._get_coef
best_svc_classifier_ridge._get_param_names
best_svc_classifier_ridge.probability
# Predicting the Test set results
y_pred_ridge =best_svc_classifier_ridge.predict(X_test_sub_ar)
# Making the Confusion Matrix and the accuracy score
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
cm = confusion_matrix(y_test_sub, y_pred_ridge)
accuracy_score(y_test_sub, y_pred_ridge).round(2)*100#80%
print(cm)#
print("Accuracy of the model emerged from a detailed grid search in SVC (Ridge) is equal to: {:.2f} %".format(
    accuracy_score(y_test_sub, y_pred_ridge).round(2)*100))
precision_score(y_test_sub, y_pred_ridge).round(2)*100#78.0
recall_score(y_test_sub, y_pred_ridge).round(2)*100#68
f1_score(y_test_sub, y_pred_ridge).round(2)*100#73.0




