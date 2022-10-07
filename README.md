Kaggle competition of Titanic classification: Accuracy score of 78.47% (this performance belongs to the top 10% in Kaggle competition)-
 Kaggle URL: https://www.kaggle.com/competitions/titanic/overview
==============================

Overview

A short description of the project. In this project our purpose is to analyse the data in order to understand what kind of passengers are more likely to survive in Titanic as well as build a classification algorithm to predict whether a passenger will survive or not. The second purpose will be evaluated in a public leaderboard test dataset of the Kaggle through the accuracy classification metric.

Concerning the first task of exploratory analysis, we examined which characteristics (features) are the most useful to extract inference about which passengers are more likely to survive. Thus, we present here the exploratory analysis results between the final selected features for our model trains and the response variable of Survival (binary) to detect meaningful patterns. The most meaningful patterns are the following ones:

- Passengers who embarked from port C are more likely to survive than the remaining ports (equal probability in the remaining two ports). 
- People who are not alone in the boat have better chances to survive.
- People with title Mrs and Miss have great chances to survive while also the title Master provides better than 50% probability of Survival.
- People who are under 17 years old have 55% probability to survive while in the remaining Age categories (16-32, 33-48, 49-64) the probability of survival are little lower than 50% and the worst chances belong to the oldest age category (>64 years old).
-An important statistical difference in terms of probability of survival is observed between two genders where the female passengers have 74% probability to survive while the male ones have ~19%.
- Finally, related to the continuous numeric features, we observe that as higher are the Fare ticket and the Passenger class of passengers, so better are the chances of survival.

As far as concerned the evaluation of our predictions, we submitted weighted averaging ensembled predictions of XGBClassifier and Random Forest classifiers obtained through nested cross-validation scheme. In this nested cross-validation scheme, we implemented hyperparameters search using optuna and Bayesian Random search for XGB Classifer and Random Forest algorithms, respectively. The best accuracy score of my own (above final solution) in Kaggle competition was 78.47% indicating that we are able to predict approximately the 80% of passengers about their survival or not. Last but not least, it is important to mention here that several classification algorithms (including XGBoost, Random Forest and Deep Learningh Keras Classifier) were applied using different combinations of features and different cross validation scheme before we resort to the application of both XGBoost and Random Forest ones as components of the ensembled solution.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed. (Not exists here)
    │   ├── processed      <- The final, canonical data sets for modeling. (Not exists here)
    │   └── raw            <- The original, immutable data dump. (Not exists here)
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details (Not exists here)
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │   ├── RF Predictions       <- Predictions in each cv outer split (nested cross-validation) using Random Forest.
    │   └── XGB Predictions      <- >> using XGB.
    │   └── ANN_KerasClassifier      <- >> using Keras Classifier (Artificial Neural Networks).
    │   └── Ensemble Predictions_XGB_RF <- >> using weighted averaging ensembling of both XGB and Random Forest classifiers.
    │   └── Ensemble Predictions_XGB_RF_Keras <- >> using weighted averaging ensembling of both XGB, Random Forest and Keras classifiers.
    │   └── Submission           <- Final submitted csv file with the predictions required for the Kaggle compeition's public leaderboard test dataset. (Kaggle Score: 78.47%)
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`. (Not exists here)
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc. (Not exists here)
    │   └── figures        <- Generated graphics and figures to be used in reporting (Not exists here)
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported (Not used in this project)
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- (Not used in this project)
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   ├   └──init__.py    <- Makes src a Python module: Here we import the libraries required for this project
    │   │   └── make_dataset.py <-Load the external data from the data/external directory
    │   │
    │   ├── features     <- Scripts to process raw data to create new features which can be evaluated using both Feature Importance algorithms and   eda     ││                                                      techniques  in order to decide the final features for modeling purposes. 
    │   │   └── build_features.py                <- Create features through several feature engineering techniques.
    │   │   └── Data_Pipelines.py                <- Data Processing Pipelines preparation in order to use them in the next scripts
    │   │   └── Feature_Importance.py            <- Feature Importance measurement using Boruta-Shap values
    │   │   └── eda.py                           <- Exploratory data analysis and visualisation to detech significant patterns
    │   │   └── final_data_train.py              <- After both 2 previous steps, decide which features will be used for the final model training
    │   │
    │   ├── Adversarial validation       <- Scripts to check potential concept/distribution shift issues (both response variable and covariates distribution shift issues)
    │   │   └── adversarial_valid.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── Optuna_Search_XGB.py   <- Hyperparameter Optimisation using Optuna method and training of XGB Classifier in each cv splits (nested cross-validation)
    │   │   └── Bayes_Search_RF.py     <- >> using Bayes method and training of RandomForest Classifier (RF) in each cv splits (nested cross-validation)
    │   │   └── Optuna_Search_ANN.py     <-Hyperparameter Optimisation using Optuna method and training of (ANN) Keras Classifier in each cv splits (nested cross-validation)
    │   │   ├── Ensemble_XGB_RF.py     <- Combine the predictions obtained using nested cross validation predictions from both XGB and RF models' trains in both previous scripts.
    │   │   ├── Ensemble_different_models_xgb_rf_keras.py     <- Combine the predictions obtained using nested cross validation predictions from both XGB,RF and Keras Classifiers trains in both previous scripts.
    │   │   ├── predict_model.py       <-Use the nested cross validated predictions of the publiv leaderboard test set to obtain a csv file for submission.
    │   │
    │   ├── Further trials             <- Additional scripts are provided here. Those ones constitute several considerations that we applied in practice beforehand of the final solution.
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

Future Research

- Apart from Stratified K-fold already used, apply techniques such as Under-Sampling or Over-Sampling to handle the issue of imbalanced datasets (some ideas were applied without providing better results but it should be implemented later).
- Apply further feature engineering to detect meaningful patterns through the error analysis of our predictions.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
