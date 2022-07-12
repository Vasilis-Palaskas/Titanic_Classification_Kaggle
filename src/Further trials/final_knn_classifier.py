#-----Îš-Nearest Neighbors Classification (K-NN)-----

from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train_sub_ar, y_train_sub)



# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
#---parts of grid search because each solver depends on the penalty chosen
#-First 
#loss_1 = ['hinge', 'squared_hinge']
#penalty_1 = ['l2']
n_neighbors_1=np.array(range(5,100,5))
weights_1=["uniform","distance"]
algorithm_1=["auto", "ball_tree", "kd_tree", "brute"]
metric_1=["minkowski","euclidean","mahalanobis",
          "chebyshev","manhattan","hamming"]


# define grid search
grid_1 = dict(n_neighbors=n_neighbors_1,
              weights=weights_1, 
              algorithm=algorithm_1,
             metric= metric_1)
#grid_2 = dict(kernel=kernel_2,
#              C=c_values_2, gamma=gamma_values_2)

#--Combine parameters to a single array (Combine different dictionairies)
knn_parameters =[grid_1]
knn_grid_search = GridSearchCV(estimator = knn_classifier,
                           param_grid = knn_parameters ,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
knn_grid_search = knn_grid_search.fit(X_train_sub_ar, y_train_sub)
best_accuracy = knn_grid_search.best_score_
best_parameters = knn_grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))#81.04 %
print("Best Parameters:", best_parameters)# {'algorithm': 'auto', 'metric': 'manhattan', 
#'n_neighbors': 10, 'weights': 'uniform'}

# Predicting the Out of sample as well as In sample predictive results based on final logistic classifier

knn_classifier = KNeighborsClassifier(algorithm="auto",metric="manhattan",n_neighbors=10,weights="uniform")
knn_classifier.fit(X_train_sub_ar, y_train_sub)

y_pred_1 = knn_classifier.predict(X_test_sub_ar)
y_train_pred_1 = knn_classifier.predict(X_train_sub_ar)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Making the Confusion Matrix and the accuracy score
cm_1 = confusion_matrix(y_test_sub, y_pred_1)

print(cm_1)
#--Accuracy score in both test and train sets as well as the f1 score in test sets
accuracy_score(y_test_sub, y_pred_1).round(2)*100  # 82.0 %%
accuracy_score(y_train_sub, y_train_pred_1).round(2)*100  # 84.0%
f1_score(y_test_sub, y_pred_1).round(2)*100  # 73.0%%


#---more detailed grid search based on the results of previous grid searches
#-First 

n_neighbors_1=np.array(range(6,14,1))
weights_1=["uniform","distance"]
algorithm_1=["auto", "ball_tree", "kd_tree", "brute"]
metric_1=["minkowski","euclidean","mahalanobis",
          "chebyshev","manhattan","hamming"]


# define grid search
grid_1 = dict(n_neighbors=n_neighbors_1,
              weights=weights_1, 
              algorithm=algorithm_1,
             metric= metric_1)
#--Combine parameters to a single array (Combine different dictionairies)
knn_parameters =[grid_1]
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train_sub_ar, y_train_sub)

knn_grid_search = GridSearchCV(estimator = knn_classifier,
                           param_grid = knn_parameters ,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
knn_grid_search_result = knn_grid_search.fit(X_train_sub_ar, y_train_sub)
best_accuracy = knn_grid_search_result.best_score_
best_parameters = knn_grid_search_result.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))#81.47 %
print("Best Parameters:", best_parameters)#{Best Parameters: {'algorithm': 'brute', 'metric': 'mahalanobis', 'n_neighbors': 9, 'weights': 'uniform'}

#--Print the best results in accuracy along with the corresponding parameter
print("Best: %2f using %2s" % (knn_grid_search_result.best_score_*100,
          knn_grid_search_result.best_params_))

best_means_5 = knn_grid_search_result.cv_results_['mean_test_score'].max()
best_means_5_index=np.where(knn_grid_search_result.cv_results_['mean_test_score']==best_means_5 )
#Print accuracy with sd
print("Best Accuracy: {:.2f} %".format(best_means_5*100))
print("Std of Accuracy: {:.2f} %".format(best_stds_5*100))

#---All means, stds and their parameters and their display
means_5 = knn_grid_search_result.cv_results_['mean_test_score']
stds_5 = knn_grid_search_result.cv_results_['std_test_score']
params_5 =knn_grid_search_result.cv_results_['params']
for mean, stdev, param in zip(means_5, stds_5, params_5):
    print("%f (%f) with: %r" % (mean, stdev, param))


#-----Final KNN results-----
#-Fitting NN using best params

best_knn_classifier_ridge =KNeighborsClassifier(n_neighbors = 9, 
                                  metric = 'mahalanobis',
                                  algorithm= 'brute',weights="uniform")

best_knn_classifier_ridge.fit(X_train_sub_ar, y_train_sub)
dir(best_knn_classifier_ridge)

# Predicting the Test set results
y_pred =best_knn_classifier_ridge.predict(X_test_sub_ar)

# Making the Confusion Matrix and the accuracy score
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
cm = confusion_matrix(y_test_sub, y_pred)
accuracy_score(y_test_sub, y_pred).round(2)*100#81.0%
print(cm)
print("Accuracy of the model emerged from a detailed grid search in SVC (Ridge) is equal to: {:.2f} %".format(
    accuracy_score(y_test_sub, y_pred).round(2)*100))#79%
precision_score(y_test_sub, y_pred).round(2)*100# 82.0
recall_score(y_test_sub, y_pred).round(2)*100#65.00
f1_score(y_test_sub, y_pred).round(2)*100# 73.0






