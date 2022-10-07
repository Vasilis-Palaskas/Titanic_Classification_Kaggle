
""" Decide which will be the final covariates/features used for the model training
"""
logger = logging.getLogger()
logger.info('After both 2 previous steps, decide which features will be used for the final model training')

# Grouping of variables to be used for our modelling purposes.
# Those skills emerged not only from our results in feature importance analysis (implemented in src/data/Feature_Importance.py)
# but also based on our exploratory analysis implemented in the script src/data/eda.py.
cat_vars=["Embarked","Sex","is_alone","title","Age_new"]
ord_vars=[]
num_vars=["sqrt_fare","Pclass"]
all_variables=["Embarked","Sex","is_alone","title","Age_new","sqrt_fare","Pclass"]
# Train, test datasets formation
new_titanic_train_data=pd.DataFrame(titanic_train_data[all_variables])
new_titanic_test_data=pd.DataFrame(titanic_test_data[all_variables])

new_titanic_train_data.head(10)
new_titanic_test_data.head(10)

X=pd.DataFrame(new_titanic_train_data[all_variables])
y=pd.DataFrame(titanic_train_data['Survived'].values)