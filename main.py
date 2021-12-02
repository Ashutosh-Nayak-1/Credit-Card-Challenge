import pandas as pd
# from sklearn.model_selection import train_test_split  (used while developing model)
# from sklearn.metrics import classification_report, confusion_matrix   (used while developing model to analyse performance)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# Reading data and dropping unnecessary columns:
data = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
data = data.drop(columns=['city', 'current_job_years', 'current_house_years', 'married'], axis=1)
test = test.drop(columns=['city', 'current_job_years', 'current_house_years', 'married'], axis=1)

# Replacing non-numeric data with dummy values in train dataset:
st = pd.get_dummies(data['state'], drop_first=False)
pr = pd.get_dummies(data['profession'], drop_first=False)
ho = pd.get_dummies(data['house_ownership'], drop_first=False)
co = pd.get_dummies(data['car_ownership'], drop_first=True)
data = data.drop(columns=['state', 'profession', 'house_ownership', 'car_ownership'], axis=1)
X = pd.concat([st, pr, ho, co, data], axis=1)

# Replacing non-numeric data with dummy values in test dataset:
t_st = pd.get_dummies(test['state'], drop_first=False)
t_pr = pd.get_dummies(test['profession'], drop_first=False)
t_ho = pd.get_dummies(test['house_ownership'], drop_first=False)
t_co = pd.get_dummies(test['car_ownership'], drop_first=True)
test = test.drop(columns=['state', 'profession', 'house_ownership', 'car_ownership'], axis=1)
X_test = pd.concat([t_st, t_pr, t_ho, t_co, test], axis=1)

# print (X.head()) (may be used to verify if above steps were executed properly)

y = X['risk_flag']    #assigning the target variable to y
x = X.drop(columns=['risk_flag'])   #removing target variable from feature matrix 

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1) (used during development of model)

# Removing Id column from train dataset and separating Id column from test dataset:
x = x.drop(columns=['Id'])
test_id = X_test['Id']
X_test = X_test.drop(columns=['Id'])

# Declaring scaler and scaling both datasets:
scale = StandardScaler().fit(x)
x = scale.transform(x)
X_test = scale.transform(X_test)

# Declaring oversampler and applying to the train dataset to obtain resampled training sets:
ovrsmp = RandomOverSampler(random_state=25)
x_resampled, y_resampled = ovrsmp.fit_resample(x, y)

# Declaring model, training it using resampled sets and applying it to make predictions on test dataset:
model=RandomForestClassifier(n_estimators=50)
model.fit(x_resampled, y_resampled)
pred = model.predict(X_test)

# Concatenating the separated Id column from earlier with predicted values, and exporting it to .csv file:
preds = pd.DataFrame({"Id":list(test_id), "risk_flag":list(pred)})
preds.to_csv("Submission.csv", index=False)

# Code used for analysis during development of model:
# print(classification_report(y_test, pred))
# print("==============================================================================")
# print("The confusion matrix is:")
# print(confusion_matrix(y_test, pred))

# =======================================================
# Miscellaneous code entered into Python Console for EDA
# =======================================================

# Necessary library imports and reading of data:
# import pandas as pd
# import matplotlib as plt
# from matplotlib import pyplot
# import seaborn as sns
# data = pd.read_csv('train.csv')

# Code to plot graph of categorical variable vs target variable:
# married=pd.crosstab(data['married'],data['risk_flag'])
# married.div(married.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True,figsize=(4,4))

# Code to plot graph of numerical variable vs target variable:
# bins=[10000, 100000, 1000000, 10000000]
# group=['poor', 'middle', 'rich']
# data['income']=pd.cut(data['income'], bins, labels=group)
# age=pd.crosstab(data['income'],data['risk_flag'])
# age.div(age.sum(1).astype(float), axis=0).plot(kind="bar",stacked=True)
# plt.xlabel('income')

# Code to generate heatmap of correlation matrix:
# data = data.drop(columns=['Id', 'profession', 'city', 'state', 'risk_flag'], axis=1)
# matrix = data.corr()
# f, ax = pyplot.subplots(figsize=(9,6))
# sns.heatmap(matrix, vmax=.8, square=True, cmap="YlGnBu", annot=True)
