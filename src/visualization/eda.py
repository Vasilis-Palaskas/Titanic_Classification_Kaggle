
###-----------------Exploratory Data Analysis (EDA)---------
""" Runs exploratory data analysis and visualisation to detect the most
        meaningful patterns in our data
"""
logger = logging.getLogger()
logger.info('Univariate analysis')

#%%%   Univariate EDA (both train and test sets)

#--- Sex 
# Creating a grid figure with matplotlib
fig, my_grid = plt.subplots(nrows=1, ncols=2, figsize=(8,6))
# 
# Plot 1
g1 =sn.countplot(x='Sex',data=titanic_train_data, palette='rainbow',ax=my_grid[0])
# Title of the Plot 1
g1.set_title("Count of Passengers' Sex (Train)")

# Plot 2
g2 =sn.countplot(x='Sex',data=titanic_test_data, palette='rainbow',ax=my_grid[1])
# Title of the Plot 2
g2.set_title("Count of Passengers' Sex (Test)")

#---Age 
# Creating a grid figure with matplotlib
fig, my_grid = plt.subplots(nrows=1, ncols=2, figsize=(8,6))
 
# Plot 1
g1 =sn.kdeplot(
    titanic_train_data['Age'],ax=my_grid[0])
# Title of the Plot 1
g1.set_title("Age (Train)")

# Plot 2
g2 =sn.kdeplot(titanic_test_data['Age'],ax=my_grid[1])
# Title of the Plot 1
g2.set_title("Age (Test)")# Seems to be approximated by the Gaussian distribution


#--- Logarithm of Age 

fig, my_grid = plt.subplots(nrows=1, ncols=2, figsize=(8,6))
# 
# Plot 1
g1 =sn.kdeplot(
    titanic_train_data['log_Age'],ax=my_grid[0])
# Title of the Plot 1
g1.set_title("log_Age(Train)")

# Plot 2
g2 =sn.kdeplot(titanic_test_data['log_Age'],ax=my_grid[1])
# Title of the Plot 1
g2.set_title("log_Age (Test)")# Seems to be approximated by the Gaussian distribution

#--- Sqrt of Age 

fig, my_grid = plt.subplots(nrows=1, ncols=2, figsize=(8,6))
# 
# Plot 1
g1 =sn.kdeplot(
    titanic_train_data['sqrt_Age'],ax=my_grid[0])
# Title of the Plot 1
g1.set_title("sqrt_Age(Train)")

# Plot 2
g2 =sn.kdeplot(titanic_test_data['sqrt_Age'],ax=my_grid[1])
# Title of the Plot 1
g2.set_title("sqrt_Age (Test)")# Seems to be approximated by the Gaussian distribution

#---Fare
fig, my_grid = plt.subplots(nrows=1, ncols=2, figsize=(8,6))

# Plot 1
g1 =sn.kdeplot(
    titanic_train_data['Fare'],ax=my_grid[0])
# Title of the Plot 1
g1.set_title("Fare (Train)")

# Plot 2
g2 =sn.kdeplot(titanic_test_data['Fare'],ax=my_grid[1])
# Title of the Plot 1
g2.set_title("Fare (Test)")# Seems to be approximated by a right skewed distribution (logarithm may fix it)

#--- Log of Fare
fig, my_grid = plt.subplots(nrows=1, ncols=2, figsize=(8,6))

# Plot 1
g1 =sn.kdeplot(
    titanic_train_data['log_fare'],ax=my_grid[0])
# Title of the Plot 1
g1.set_title("Logarithm of Fare (Train)")

# Plot 2
g2 =sn.kdeplot(
    titanic_test_data['log_fare'],ax=my_grid[1])
# Title of the Plot 1
g2.set_title("Logarithm of Fare (Test)")

#--- Sqrt of Fare
fig, my_grid = plt.subplots(nrows=1, ncols=2, figsize=(8,6))

# Plot 1
g1 =sn.kdeplot(
    titanic_train_data['sqrt_fare'],ax=my_grid[0])
# Title of the Plot 1
g1.set_title("Sqrt of Fare (Train)")

# Plot 2
g2 =sn.kdeplot(
    titanic_test_data['sqrt_fare'],ax=my_grid[1])
# Title of the Plot 1
g2.set_title("S of Fare (Test)")

#--- Pclass

fig, my_grid = plt.subplots(nrows=1, ncols=2, figsize=(8,6))
# 
# Plot 1
g1 =sn.countplot(x='Pclass',data=titanic_train_data, palette='rainbow',ax=my_grid[0])
# Title of the Plot 1
g1.set_title("Ticket class Frequencies of Passengers (Train)")

# Plot 2
g2 =sn.countplot(x='Pclass',data=titanic_test_data, palette='rainbow',ax=my_grid[1])
# Title of the Plot 2
g2.set_title("Ticket class Frequencies of Passengers (Test)")


#--- Embarked

fig, my_grid = plt.subplots(nrows=1, ncols=2, figsize=(8,6))
# 
# Plot 1
g1 =sn.countplot(x='Embarked',data=titanic_train_data, palette='rainbow',ax=my_grid[0])
# Title of the Plot 1
g1.set_title("Embarked of Passengers (Train)")

# Plot 2
g2 =sn.countplot(x='Embarked',data=titanic_test_data, palette='rainbow',ax=my_grid[1])
# Title of the Plot 2
g2.set_title("Embarked of Passengers (Test)")

#--- Is alone

fig, my_grid = plt.subplots(nrows=1, ncols=2, figsize=(8,6))
#
# Plot 1
g1 =sn.countplot(x='is_alone',data=titanic_train_data, palette='rainbow',ax=my_grid[0])
# Title of the Plot 1
g1.set_title("# of Passengers being alone or not in the ship (Train)")

# Plot 2
g2 =sn.countplot(x='is_alone',data=titanic_test_data, palette='rainbow',ax=my_grid[1])
# Title of the Plot 2
g2.set_title("# of Passengers being alone or not in the ship (Test)")

#%%%   Bivariate EDA (both train and test sets)
logger.info('Bivariate analysis')
#---Prob. of our binary response variable (Survived) across the levels of categorical variables

print("----------------------")
print(titanic_train_data[['is_alone','Survived']].groupby(['is_alone'], as_index = False).mean())# Prob. of survived depending on whether someone is alone or not 
print(titanic_train_data[['family_size','Survived']].groupby(['family_size'], as_index = False).mean())#>> & family size
print(titanic_train_data[['title','Survived']].groupby(['title'], as_index = False).mean())## >> depending on the title of each one
print(titanic_train_data[["category_fare","Survived"]].groupby(["category_fare"], as_index = False).mean() )#>> depending on the Fare category
print( titanic_train_data[["category_age","Survived"]].groupby(["category_age"], as_index = False).mean() )#>> depending on the Age category
print( titanic_train_data[["Embarked","Survived"]].groupby(["Embarked"], as_index = False).mean() )#>> depending on the "Embarked station
print( titanic_train_data[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean() ))#>> depending on the Pclass
print("----------------------")


#--Correlations Pearson between features and Response for the train set
corr_matrix_train =titanic_train_data.corr()
corr_matrix_train
corr_matrix_train["Survived"]

#-------Stacked barplots for the bivariate analysis (only for train because in train set we have available the resonse)

#-PClass-Survived
fig, my_grid = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
# Plot 1
g1 =sn.countplot(x='Pclass',data=titanic_train_data, palette='rainbow',hue='Survived')# Concerning the 3rd class' passengers, the majority of them did not survive.
# Title of the Plot 1
g1.set_title("Count of Passengers that Survived, Separated by Ticket Class (Train)")# Only in 1st class'passengers, the proportion of them who survived was greater than the one of them who did not survive.

# Is alone-Survived
fig, my_grid = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
# Plot 1
g1 =sn.countplot(x='is_alone',data=titanic_train_data, palette='rainbow',hue='Survived')# Concerning the 3rd class' passengers, the majority of them did not survive.
# Title of the Plot 1
g1.set_title("Count of Passengers that Survived, depending on whether they were or not alone (Train)")# Only in 1st class'passengers, the proportion of them who survived was greater than the one of them who did not survive.

#-Sex-Survived
plt.figure(figsize=(8,5))
# PClass-Survived
fig, my_grid = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
# Plot 1
g1 =sn.countplot(x='Sex',data=titanic_train_data, palette='rainbow',hue='Survived')# Male mostly did not survive and females mostly survive.Significant difference
# Title of the Plot 1
g1.set_title("Count of Passengers that Survived, Separated by Sex (Train)")# 

#-Embarked-Survived
plt.figure(figsize=(8,5))
fig, my_grid = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
# Plot 1
g1 =sn.countplot(x='Embarked',data=titanic_train_data, palette='rainbow',hue='Survived')#Embarked S did not survive while C survived. Q did not survive mostly.Significant difference
# Title of the Plot 1
g1.set_title("Count of Passengers that Survived, Separated by Embarked (Train)")# 

# Title-Survived
plt.figure(figsize=(8,5))
# PClass-Survived
fig, my_grid = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
# Plot 1
g1 =sn.countplot(x='title',data=titanic_train_data, palette='rainbow',hue='Survived')#Embarked S did not survive while C survived. Q did not survive mostly.Significant difference
# Title of the Plot 1
g1.set_title("Count of Passengers that Survived, Separated by their Title (Train)")# 

#-Age category-Survived
plt.figure(figsize=(8,5))
fig, my_grid = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
# Plot 1
g1 =sn.countplot(x='Age_new',data=titanic_train_data, palette='rainbow',hue='Survived')#Embarked S did not survive while C survived. Q did not survive mostly.Significant difference
# Title of the Plot 1
g1.set_title("Count of Passengers that Survived, Separated by their Age category (Train)")# 

#-Fare_new-Survived
plt.figure(figsize=(8,5))
#Fare-Survived
fig, my_grid = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
# Plot 1
g1 =sn.countplot(x='Fare_new',data=titanic_train_data, palette='rainbow',hue='Survived')#Embarked S did not survive while C survived. Q did not survive mostly.Significant difference
# Title of the Plot 1
g1.set_title("Count of Passengers that Survived, Separated by their Fare category (Train)")# 



#-- Boxplots (Bivariate Relationships)

#- Ticket Fare-Survived
fig, my_grid = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
#
# Plot 1
g1 =sn.boxplot(x='Survived',y='Fare',data=titanic_train_data, palette='rainbow')
# Title of the Plot 1
g1.set_title("Ticket Fare by Passenger Survived, Titanic (Train)")# The more expensive is the ticket fare, the more probable is someone to survive

#- Sqrt of Ticket Fare-Survived
fig, my_grid = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
#
# Plot 1
g1 =sn.boxplot(x='Survived',y='sqrt_fare',data=titanic_train_data, palette='rainbow')
# Title of the Plot 1
g1.set_title("Sqrt Ticket Fare by Passenger Survived, Titanic (Train)")# The more expensive is the ticket fare, the more probable is someone to survive


#- Age-Survived
fig, my_grid = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
#
# Plot 1
g1 =sn.boxplot(x='Survived',y='Age',data=titanic_train_data, palette='rainbow')
# Title of the Plot 1wr
g1.set_title("Age by Passenger Survived, Titanic (Train)")# Not so much difference is observed between ages and prob. of Survival


#-Ticket Fare-Pclass
fig, my_grid = plt.subplots(nrows=1, ncols=2, figsize=(8,6))
#
# Plot 1
g1 =sn.boxplot(x='Pclass',y='Fare',data=titanic_train_data, palette='rainbow',ax=my_grid[0])
# Title of the Plot 1
g1.set_title("Ticket Fare by Passenger Class, Titanic (Train)")

#-Plot 2
g2 =sn.boxplot(x='Pclass',y='Fare',data=titanic_test_data, palette='rainbow',ax=my_grid[1])
# Title of the Plot 2
g2.set_title("Ticket Fare by Passenger Class, Titanic (Test)")

#-Ticket Fare-Embarked Port

fig, my_grid = plt.subplots(nrows=1, ncols=2, figsize=(8,6))
#
# Plot 1
g1 =sn.boxplot(x='Embarked',y='Fare',data=titanic_train_data, palette='rainbow',ax=my_grid[0])
# Title of the Plot 1
g1.set_title("Ticket Fare by Passenger Embarked, Titanic (Train)")

# Plot 2
g2 =sn.boxplot(x='Embarked',y='Fare',data=titanic_test_data, palette='rainbow',ax=my_grid[1])
# Title of the Plot 2
g2.set_title("Ticket Fare by Passenger Embarked, Titanic (Test)")
