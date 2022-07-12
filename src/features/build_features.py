
#%%%% For both train and test sets, feature engineering
""" Create new features which are meaningful for our case
"""
logger = logging.getLogger()
logger.info('New features creation')
#--Feature 1: Size of family
titanic_train_data['family_size'] = titanic_train_data['SibSp'] + titanic_train_data['Parch'] + 1
titanic_test_data['family_size'] = titanic_test_data['SibSp'] + titanic_test_data['Parch'] + 1

#Feature 1.1: Is alone or Not
titanic_train_data['is_alone'] = 0
titanic_train_data.loc[titanic_train_data['family_size'] == 1, 'is_alone'] = 1

titanic_test_data['is_alone'] = 0
titanic_test_data.loc[titanic_test_data['family_size'] == 1, 'is_alone'] = 1
# Prob. of survived depending on whether someone is alone or not.
print(titanic_train_data[['is_alone','Survived']].groupby(['is_alone'], as_index = False).mean())
print(titanic_train_data[['family_size','Survived']].groupby(['family_size'], as_index = False).mean())

#--Feature 2: Title
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\. ', name)
    if title_search:
        return title_search.group(1)
    return ""

# Firstly convert them to string
titanic_train_data['Name']= titanic_train_data['Name'].astype('str') 
titanic_test_data['Name']= titanic_test_data['Name'].astype('str') 
# apply function of fer_title fir both train and test sets

titanic_train_data['title'] = titanic_train_data['Name'].apply(get_title)
titanic_test_data['title'] = titanic_test_data['Name'].apply(get_title)


titanic_train_data['title'] = titanic_train_data['title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
titanic_train_data['title'] = titanic_train_data['title'].replace('Mlle','Miss')
titanic_train_data['title'] = titanic_train_data['title'].replace('Ms','Miss')
titanic_train_data['title'] =titanic_train_data['title'].replace('Mme','Mrs')    
    
titanic_test_data['title'] = titanic_test_data['title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],'Rare')
titanic_test_data['title'] = titanic_test_data['title'].replace('Mlle','Miss')
titanic_test_data['title'] = titanic_test_data['title'].replace('Ms','Miss')
titanic_test_data['title'] =titanic_test_data['title'].replace('Mme','Mrs')   

#--Feature 3: Normalisation and sqrt of Fare ticket

# 3.1: Logarithm
titanic_train_data["log_fare"]=titanic_train_data["Fare"]
titanic_test_data["log_fare"]=titanic_test_data["Fare"]
#Proper action to avoid -Inf values when apply log
titanic_train_data.loc[titanic_train_data["log_fare"]==0,"log_fare"]=0.1
titanic_test_data.loc[titanic_test_data["log_fare"]==0,"log_fare"]=0.1

titanic_train_data["log_fare"]=np.log10(titanic_train_data["log_fare"])
titanic_test_data["log_fare"]=np.log10(titanic_test_data["log_fare"])

# 3.2:Sqrt
titanic_train_data["sqrt_fare"]=titanic_train_data["Fare"]
titanic_test_data["sqrt_fare"]=titanic_test_data["Fare"]

titanic_train_data["sqrt_fare"]=np.sqrt(titanic_train_data["sqrt_fare"])
titanic_test_data["sqrt_fare"]=np.sqrt(titanic_test_data["sqrt_fare"])
#Feature 3.3 Fare-Groupping
titanic_train_data['category_fare'] = pd.qcut(titanic_train_data['Fare'], 4)
titanic_test_data['category_fare'] = pd.qcut(titanic_test_data['Fare'], 4)

#Mapping Fare based on the binnning implemented above
titanic_train_data['Fare_new']=titanic_train_data['Fare']# create new identical feature to avoid confusions 
titanic_test_data['Fare_new']=titanic_test_data['Fare'] #>>

titanic_train_data.loc[ titanic_train_data['Fare'] <= 7.91, 'Fare_new'] = 0
titanic_train_data.loc[(titanic_train_data['Fare'] > 7.91) & (titanic_train_data['Fare'] <= 14.454), 'Fare_new'] = 1
titanic_train_data.loc[(titanic_train_data['Fare'] > 14.454) & (titanic_train_data['Fare'] <= 31), 'Fare_new']   = 2
titanic_train_data.loc[ titanic_train_data['Fare'] > 31, 'Fare_new']                               = 3
titanic_train_data['Fare_new'] = titanic_train_data['Fare_new'].astype(int)

titanic_test_data.loc[ titanic_test_data['Fare'] <= 7.91, 'Fare_new']                            = 0
titanic_test_data.loc[(titanic_test_data['Fare'] > 7.91) & (titanic_test_data['Fare'] <= 14.454), 'Fare_new'] = 1
titanic_test_data.loc[(titanic_test_data['Fare'] > 14.454) & (titanic_test_data['Fare'] <= 31), 'Fare_new']   = 2
titanic_test_data.loc[ titanic_test_data['Fare'] > 31, 'Fare_new']                               = 3

#--Feature 4: Normalisation and sqrt of Age

#4.1: Logarithm age

titanic_train_data["log_Age"]=titanic_train_data["Age"]
titanic_test_data["log_Age"]=titanic_test_data["Age"]

titanic_train_data["log_Age"]=np.log10(titanic_train_data["log_Age"])
titanic_test_data["log_Age"]=np.log10(titanic_test_data["log_Age"])
#4.2:  Sqrt age
titanic_train_data["sqrt_Age"]=titanic_train_data["Age"]
titanic_test_data["sqrt_Age"]=titanic_test_data["Age"]

titanic_train_data["sqrt_Age"]=np.sqrt(titanic_train_data["sqrt_Age"])
titanic_test_data["sqrt_Age"]=np.sqrt(titanic_test_data["sqrt_Age"])

#Feature 4.3 Age-Groupping

age_avg  = titanic_train_data['Age'].mean()
age_std  = titanic_train_data['Age'].std()
age_null_train = titanic_train_data['Age'].isnull().sum()
age_null_test = titanic_test_data['Age'].isnull().sum()

random_list_train = np.random.randint(age_avg - age_std, age_avg + age_std , size = age_null_train)
random_list_test = np.random.randint(age_avg - age_std, age_avg + age_std , size = age_null_test)

# Age categories (replace missing values of Age)

titanic_train_data['Age_cat']=titanic_train_data['Age']
titanic_train_data['Age_cat'][np.isnan(titanic_train_data['Age'])] = random_list_train
titanic_train_data['Age_cat'] = titanic_train_data['Age_cat'].astype(int)

titanic_test_data['Age_cat']=titanic_test_data['Age']
titanic_test_data['Age_cat'][np.isnan(titanic_test_data['Age'])] =random_list_test
titanic_test_data['Age_cat'] = titanic_test_data['Age_cat'].astype(int)
# Age category grouping
titanic_train_data['category_age'] = pd.cut(titanic_train_data['Age_cat'], 5)
titanic_test_data['category_age'] = pd.cut(titanic_test_data['Age_cat'], 5)

# Mapping Age based on the binnning implemented above
titanic_train_data['Age_new']= titanic_train_data['Age']# create new identical feature to avoid confusion
titanic_test_data['Age_new']= titanic_test_data['Age']#>>

titanic_train_data.loc[ titanic_train_data['Age'] <= 16, 'Age_new']= 0
titanic_train_data.loc[(titanic_train_data['Age'] > 16) & (titanic_train_data['Age'] <= 32), 'Age_new'] = 1
titanic_train_data.loc[(titanic_train_data['Age'] > 32) & (titanic_train_data['Age'] <= 48), 'Age_new'] = 2
titanic_train_data.loc[(titanic_train_data['Age'] > 48) & (titanic_train_data['Age'] <= 64), 'Age_new'] = 3
titanic_train_data.loc[titanic_train_data['Age'] > 64, 'Age_new']    = 4

titanic_test_data.loc[ titanic_test_data['Age'] <= 16, 'Age_new']= 0
titanic_test_data.loc[(titanic_test_data['Age'] > 16) & (titanic_test_data['Age'] <= 32), 'Age_new'] = 1
titanic_test_data.loc[(titanic_test_data['Age'] > 32) & (titanic_test_data['Age'] <= 48), 'Age_new'] = 2
titanic_test_data.loc[(titanic_test_data['Age'] > 48) & (titanic_test_data['Age'] <= 64), 'Age_new'] = 3
titanic_test_data.loc[ titanic_test_data['Age'] > 64, 'Age_new']    = 4


#-- Dropping of several useless variables for both train and test datasets
columns_to_drop = ["PassengerId","Ticket", "Cabin","Name", "SibSp", "Parch", "family_size","category_fare","category_age","Age_cat"]
titanic_train_data=titanic_train_data.drop(columns_to_drop, axis=1)
titanic_test_data=titanic_test_data.drop(columns_to_drop, axis=1)
# Check whether missing values exist or not
titanic_train_data.isna().sum().sort_values(ascending=False)#age and embarked include missing values
titanic_test_data.isna().sum().sort_values(ascending=False)# age and fare >>
