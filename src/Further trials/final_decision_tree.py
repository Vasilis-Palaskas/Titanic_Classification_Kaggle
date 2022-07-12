

# -----Decision Tree Classification-----
# Fitting DTC to the Training set
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier(random_state=0)
tree_clf.fit(X_train_sub_ar, y_train_sub)


# Applying Grid Search to find the best model and the best parameters
# ---parts of grid search because each solver depends on the penalty chosen
# -First

criterion_1 = ['gini', "entropy"]
splitter_1 = ["best"]
# The minimum number of samples required to split an internal node:
min_samples_split_1 = np.array(range(2, 41, 3))
min_samples_leaf_1 = np.array(range(1, 21, 2))
max_features_1 = ["log2", "auto","sqrt"]
max_depth_1 =  np.array(range(2, 15, 1))


# define grid search
grid_1 = dict(criterion=criterion_1, splitter=splitter_1,
              min_samples_split=min_samples_split_1,
              min_samples_leaf=min_samples_leaf_1,
              max_features=max_features_1,max_depth=max_depth_1)

# --Combine parameters to a single array (Combine different dictionairies)
dct_parameters = [grid_1]
dct_grid_search = GridSearchCV(estimator=tree_clf,
                               param_grid=dct_parameters,
                               scoring='accuracy',
                               cv=10,
                               n_jobs=-1)
dct_grid_search_result = dct_grid_search.fit(X_train_sub_ar, y_train_sub)


best_accuracy = dct_grid_search_result.best_score_
best_parameters = dct_grid_search_result.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))#83.01 %
print("Best Parameters:", best_parameters)# { {'criterion': 'entropy', 'max_depth': 5, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 5, 'splitter': 'best'}


# ---Final tree

tree_clf_after_grid_search = DecisionTreeClassifier(criterion='entropy',
                                                    max_features='log2', 
                                                    min_samples_leaf=1,max_depth=5,
                                                    min_samples_split=5, splitter='best')
tree_clf_after_grid_search_result = tree_clf_after_grid_search.fit(
    X_train_sub_ar, y_train_sub)
tree_clf_after_grid_search_result_with_scaling = tree_clf_after_grid_search.fit(
    X_train_sub_ar, y_train_sub)

y_pred_1 =tree_clf_after_grid_search_result.predict(X_test_sub_ar)
y_train_pred_1 = tree_clf_after_grid_search_result.predict(X_train_sub_ar)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Making the Confusion Matrix and the accuracy score
cm_1 = confusion_matrix(y_test_sub, y_pred_1)

print(cm_1)
#--Accuracy score in both test and train sets as well as the f1 score in test sets
accuracy_score(y_test_sub, y_pred_1).round(2)*100  # 82.0 %%
accuracy_score(y_train_sub, y_train_pred_1).round(2)*100  # 84.0%

f1_score(y_test_sub, y_pred_1).round(2)*100  # 72.0%%


# ---more detailed grid search based on the initial grid search:# {'criterion': 'gini', 'max_depth': 14, 'max_features': 'log2', 'min_samples_leaf': 7, 'min_samples_split': 2, 'splitter': 'best'}
criterion_1 = ["entropy", "gini"]
splitter_1 = ["best"]
min_samples_split_1 = np.array(range(2, 10, 1))
min_samples_leaf_1 = np.array(range(1, 7, 1))
max_features_1 = ["log2"]
max_depth_1=np.array(range(4, 9, 1))
# define grid search
final_grid = dict(criterion=criterion_1, splitter=splitter_1,
                  min_samples_split=min_samples_split_1,
                  min_samples_leaf=min_samples_leaf_1,
                  max_features=max_features_1,max_depth=max_depth_1)

# --Combine parameters to a single array (Combine different dictionairies)
dct_parameters = [final_grid]
final_dct_grid_search = GridSearchCV(estimator=tree_clf,
                                     param_grid=dct_parameters,
                                     scoring='accuracy',
                                     cv=10,
                                     n_jobs=-1)
#final_dct_grid_search_result = final_dct_grid_search.fit(
#    X_train_sub_ar, y_train_sub)
final_dct_grid_search_result = final_dct_grid_search.fit(
    X_train_sub_ar, y_train_sub)

best_accuracy = final_dct_grid_search_result.best_score_
best_parameters = final_dct_grid_search_result.best_params_

print("Best Accuracy: {:.2f} %".format(best_accuracy*100))  # 83.01
print("Best Parameters:", best_parameters)#{'criterion': 'gini', 'max_depth': 14, 'max_features': 'log2', 'min_samples_leaf': 7, 'min_samples_split': 2, 'splitter': 'best'}
best_mean = final_dct_grid_search_result.cv_results_['mean_test_score'].max()
best_means_index=np.where(final_dct_grid_search_result.cv_results_['mean_test_score']==best_mean )
best_std=final_dct_grid_search_result.cv_results_['std_test_score'][best_means_index][0]



# ---Final tree

tree_clf_after_grid_search = DecisionTreeClassifier(criterion='entropy',
                                                    max_features='log2', 
                                                    min_samples_leaf=1,max_depth=5,
                                                    min_samples_split=5, splitter='best',random_state=0)
tree_clf_after_grid_search_result = tree_clf_after_grid_search.fit(
    X_train_sub_ar, y_train_sub)

# ---Visualisation of the Decision tree

fig = plt.figure(figsize=(35, 20))
_ = tree.plot_tree(tree_clf_after_grid_search,
                   rounded=True,
                   feature_names=X_train_sub.columns,
                   filled=True)
fig.savefig("titanic_decistion_tree.png")
#--Obtain predictions for both in and out of sample observations
y_pred_1 =tree_clf_after_grid_search_result.predict(X_test_sub_ar)
y_train_pred_1 = tree_clf_after_grid_search_result.predict(X_train_sub_ar)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Making the Confusion Matrix and the accuracy score
cm_1 = confusion_matrix(y_test_sub, y_pred_1)

print(cm_1)
#--Accuracy score in both test and train sets as well as the f1 score in test sets
accuracy_score(y_test_sub, y_pred_1).round(2)*100  # 82.0 %%
accuracy_score(y_train_sub, y_train_pred_1).round(2)*100  # 84.0%

#--Another score metrics
precision_score(y_test_sub, y_pred_1).round(2)*100  # 79.0 %%
recall_score(y_test_sub, y_pred_1).round(2)*100  # 72.0 %%

f1_score(y_test_sub, y_pred_1).round(2)*100  # 76.0%%
