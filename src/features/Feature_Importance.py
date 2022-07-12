

""" Runs Feature Importance measurement using Boruta-Shap values
"""
logger = logging.getLogger()
logger.info('Feature Importance measurement using Boruta-Shap values')
#%% Data Processing in order to apply Feature importance to the most menaingful features

model = XGBClassifier(eval_metric="logloss")# XGB Classifier

titanic_train_data.columns# print the features existing now in the train dataset
columns_to_drop = ["Survived"]# useless variables (some of them are duplicated and for this reason they are removed)

# Grouping of variables per structure
cat_vars=["Embarked","Sex","is_alone","title"]
ord_vars=["Age_new","Fare_new"]
num_vars=["Age","sqrt_Age","log_Age","Fare","sqrt_fare","log_fare","Pclass"]
all_variables=["Embarked","Sex","is_alone","title","Age_new","Fare_new",
               "Age","sqrt_Age","log_Age","Fare","sqrt_fare","log_fare","Pclass"]


#%%% Feature Importance application and data processing pipelines

#---Data processing to create datasets for the feature importance fitting
X=pd.DataFrame(titanic_train_data.drop(columns_to_drop, axis=1))
X=X[all_variables]
#y=pd.DataFrame(titanic_train_data['Survived'].values)
y=np.array(titanic_train_data['Survived'].values)


#--Data processing pipelines for each category of variables (depending on its structure)
cat_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent') ),
                            ('encoder', OneHotEncoder(handle_unknown='ignore',drop = 'first'))])

ord_pipeline=Pipeline([('imputer', SimpleImputer(strategy='most_frequent') ),
                            ('encoder', OrdinalEncoder())])

num_pipeline= Pipeline( [('scaler',  StandardScaler())] )

# Combine sub-pipelines into a single one
cat_ord_pipeline = ColumnTransformer([("cat_pipeline", cat_pipeline, cat_vars),
                               ("ord_pipeline", ord_pipeline, ord_vars)],
                                  remainder="passthrough")
cat_ord_num_pipeline = ColumnTransformer([("cat_pipeline", cat_pipeline, cat_vars),
                               ("ord_pipeline", ord_pipeline, ord_vars),
                               ("num_pipeline", num_pipeline, num_vars)]
                                  )
# Define which will be the final variables to be tested and the corresponding datasets for F.S.

all_variables_with_ordinal=["Embarked_Q","Embarked_S","Sex_male","is_alone",
               "title_Miss","title_Mr","title_Mrs","title_Rare","Age_new","Fare_new",
               "Age","sqrt_Age","log_Age","Fare","sqrt_fare","log_fare","Pclass"]
#-Apply the data processing pipelines in our X dataset
X_cat_ord_pipeline_with_ordinal=pd.DataFrame(cat_ord_pipeline.fit_transform(X),columns=all_variables_with_ordinal)
X_cat_ord_num_pipeline_with_ordinal=pd.DataFrame(cat_ord_num_pipeline.fit_transform(X),columns=all_variables_with_ordinal)
#--Application of Feature Selection method (no model selected default is Random Forest, if classification is False it is a Regression problem)
Feature_Selector = BorutaShap(model=model,
                              importance_measure='shap',
                              classification=True)
#---Case 1: All variables without scaling in numeric ones

Feature_Selector.fit(X=X_cat_ord_pipeline_with_ordinal, y=y, n_trials=150, random_state=0)
# Returns Boxplot of features
Feature_Selector.plot(X_size=len(all_variables_with_ordinal), figsize=(len(all_variables_with_ordinal),8),
            y_scale='log', which_features='all')#4 attributes confirmed important: ['Sex_male', 'Pclass', 'Age', 'title_Mr']

#---Case 2: All variables with scaling in numeric ones
Feature_Selector.fit(X=X_cat_ord_num_pipeline_with_ordinal, y=y, n_trials=150, random_state=0)
# Returns Boxplot of features
Feature_Selector.plot(X_size=len(all_variables_with_ordinal), figsize=(len(all_variables_with_ordinal),8),
            y_scale='log', which_features='all')#4 attributes confirmed important: ['Sex_male', 'Pclass', 'Age', 'title_Mr']

#---Case 3: All variables with scaling in numeric ones

# Here we do not consider ordinal both covariates of Age_new, Fare_new
cat_vars=["Embarked","Sex","is_alone","title","Age_new","Fare_new"]
ord_vars=[]
num_vars=["Age","sqrt_Age","log_Age","Fare","sqrt_fare","log_fare","Pclass"]
all_variables=["Embarked","Sex","is_alone","title","Age_new","Fare_new",
               "Age","sqrt_Age","log_Age","Fare","sqrt_fare","log_fare","Pclass"]


all_variables_without_ordinal=["Embarked_Q","Embarked_S","Sex_male","is_alone",
               "title_Miss","title_Mr","title_Mrs","title_Rare","Age_new_1","Age_new_2",
               "Age_new_3","Age_new_4","Fare_new_1","Fare_new_2","Fare_new_3",
               "Age","sqrt_Age","log_Age","Fare","sqrt_fare","log_fare","Pclass"]

cat_num_pipeline = ColumnTransformer([("cat_pipeline", cat_pipeline, cat_vars),
                               ("num_pipeline", num_pipeline, num_vars)]
                                  )
#-Apply the data processing pipelines in our X dataset
X_cat_num_pipeline_without_ordinal=pd.DataFrame(cat_num_pipeline.fit_transform(X),columns=all_variables_without_ordinal)
# no model selected default is Random Forest, if classification is False it is a Regression problem
Feature_Selector_without_ordinal = BorutaShap(model=model,
                              importance_measure='shap',
                              classification=True)

Feature_Selector_without_ordinal.fit(X=X_cat_num_pipeline_without_ordinal, y=y, n_trials=150, random_state=0)
# Returns Boxplot of features
Feature_Selector.plot(X_size=len(all_variables_without_ordinal), figsize=(len(all_variables_without_ordinal),8),
            y_scale='log', which_features='all')#4 attributes confirmed important: ['Sex_male', 'Pclass', 'Age', 'title_Mr']

