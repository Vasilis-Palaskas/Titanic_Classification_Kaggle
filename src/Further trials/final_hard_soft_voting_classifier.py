#----Ensemble learning paradigm

#---1) Hard voting classifier

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

#%%---Here we combine the best versions of each classifier (emerged after a detailed grid search in each one of them)
#-Fitting Logistic regression using ridge penalty L2regression
best_Logistic_classifier_ridge =  LogisticRegression(
    random_state = 0,penalty="l2",
             solver='liblinear',C=0.1)
best_Logistic_classifier_ridge.fit(X_train_sub_ar, y_train_sub)

#-Fitting SVC using best params
best_svc_classifier_ridge =SVC(C=2.2,gamma=0.1,
                               kernel="rbf",
                               random_state = 0)
best_svc_classifier_ridge.fit(X_train_sub_ar, y_train_sub)
#--Fitting SVC Classifiers With probability of each observation (such as Logistic, or Decision tree)
best_svc_classifier_ridge_prob =SVC(C=2.2,gamma=0.1,
                               kernel="rbf",probability=True,
                               random_state = 0)
best_svc_classifier_ridge_prob.fit(X_train_sub_ar, y_train_sub)
#--Best decision tree
best_dct_classifier_ridge = DecisionTreeClassifier(criterion='entropy',
                                                    max_features='log2', 
                                                    min_samples_leaf=1,max_depth=5,
                                                    min_samples_split=5, splitter='best',random_state=0)
best_dct_classifier_ridge.fit(X_train_sub_ar, y_train_sub)

#-Fitting NN using best params

best_knn_classifier_ridge =KNeighborsClassifier(n_neighbors = 9, 
                                  metric = 'mahalanobis',
                                  algorithm= 'brute',weights="uniform")

best_knn_classifier_ridge.fit(X_train_sub_ar, y_train_sub)
dir(best_knn_classifier_ridge)
#%%---

#%%%---Hard and Soft Voting classifiers
###----------lets create both hard and soft classifiers

voting_hard_clf = VotingClassifier(
    estimators=[('lr', best_Logistic_classifier_ridge),
            ('dct', best_dct_classifier_ridge),
            ('svc', best_svc_classifier_ridge),
             ('knn',  best_knn_classifier_ridge)],
                    voting='hard')

voting_soft_clf = VotingClassifier(
    estimators=[('lr', best_Logistic_classifier_ridge),
            ('dct', best_dct_classifier_ridge),
            ('svc_prob', best_svc_classifier_ridge_prob)],
                    voting='soft',n_jobs=-1)
#---Fittinf of both hard and soft classifiers
voting_hard_clf.fit(X_train_sub_ar, y_train_sub)
voting_soft_clf.fit(X_train_sub_ar, y_train_sub)

#---Compare and print the results of individual hard classifiers and their unified classifier
from sklearn.metrics import accuracy_score,f1_score
for clf in (best_Logistic_classifier_ridge,
            best_dct_classifier_ridge, 
            best_svc_classifier_ridge,best_knn_classifier_ridge,
            voting_hard_clf):
    clf.fit(X_train_sub_ar, y_train_sub)
    y_pred = clf.predict(X_test_sub_ar)
    print(clf.__class__.__name__, accuracy_score(y_test_sub, y_pred))#0.815 unifued, 0.82 decision tree
#---Compare and print the results of individual soft classifiers and their unified classifier

for clf in (best_Logistic_classifier_ridge,best_dct_classifier_ridge,
            best_svc_classifier_ridge_prob,voting_soft_clf):
    clf.fit(X_train_sub_ar, y_train_sub)
    y_pred = clf.predict(X_test_sub_ar)
    print(clf.__class__.__name__, accuracy_score(y_test_sub, y_pred))#0.81 unified, 0.82 decision tree
#---Conclusion: # 0.81 worse than DCT which is the best but better than the remaining ones
#%%%---