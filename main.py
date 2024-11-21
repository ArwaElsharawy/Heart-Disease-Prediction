import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import load_iris
from itertools import chain
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.metrics import ConfusionMatrix,plot_ConfusionMatrix
scale = StandardScaler()

data = pd.read_csv("D:\Heart_Disease.csv")

print("Num of duplicated records:")
print(sum(data.duplicated(subset="id")))
print("---------------------")

print("Num of null cells in each column:")
print(data.isnull().sum())
data.dropna(axis=0, inplace=True)
print("---------------------")

print("Num of null cells in each column after:")
print(data.isnull().sum())
print("---------------------")

data['Heart Disease']=data['Heart Disease'].replace("Yes",1)
data['Heart Disease']=data['Heart Disease'].replace("No",0)
data['Gender']=data['Gender'].replace("Female",1)
data['Gender']=data['Gender'].replace("Male",0)
#data['work_type']=data['work_type'].replace(["Govt_job","Never_worked","Private", 'Self-employed', 'children'],[0,1,2,3,4])
#data['smoking_status']=data['smoking_status'].replace(['formerly smoked', 'never smoked', 'smokes', 'Unknown'],[0,1,2,3])


data["work_type"].unique()
data["smoking_status"].unique()
ohe =OneHotEncoder()
print(ohe)
ohe.fit_transform(data[["work_type", "smoking_status"]]).toarray()
feature_arry = ohe.fit_transform(data[["work_type", "smoking_status"]]).toarray()
ohe.categories_
feature_labels = ohe.categories_
feature_labels
feature_labels=list(chain.from_iterable(feature_labels))
feature_labels
feature_labels[:]=['work_type_Govt_job', 'work_type_Never_worked','work_type_Private','work_type_Self-employed','work_type_children','smoking_status_Unknown','smoking_status_formerly smoked','smoking_status_never smoked','smoking_status_smokes']
print(feature_labels)
pd.DataFrame(feature_arry,columns= feature_labels)
features = pd.DataFrame(feature_arry, columns = feature_labels)
print(features)
data = pd.concat([data.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
data = data.drop([ 'work_type', 'smoking_status'],  axis=1)
data.dropna(axis=0, inplace=True)
data.info()


# function to find Outlier

def outlier(data, col_name):
    Q1 = np.percentile(data[col_name], 25)

    Q3 = np.percentile(data[col_name], 75)

    IQR = Q3 - Q1
    Ubound = Q3 + (1.5 * IQR)
    Lbound = Q1 - (1.5 * IQR)
    data[col_name] = np.where(data[col_name] < Lbound, Lbound, data[col_name])
    data[col_name] = np.where(data[col_name] > Ubound, Ubound, data[col_name])

sns.boxplot(data,palette='rainbow',orient='h')

for i in data.columns:
  outlier(data,i)

sns.boxplot(data,palette='rainbow',orient='h')

sns.heatmap(data.corr())
plt.show()

data = data.drop(['id'], axis=1)

scaler = MinMaxScaler()
scaler.fit(data)
scaled = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled, columns=data.columns)
print(scaled_df)
data=scaled_df

target=data['Heart Disease']
data=data.drop("Heart Disease",axis=1)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=1) # 70% training and 30% test


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy MI :",metrics.accuracy_score(y_test, y_pred))


#Chi_square Test

form sklearn.feature_selection import selectKBest
from sklearn.feature_selection import chi2

X_cat =data.astype(int)
chi2_feature = selectKBest(chi2,k=3)
X_Kbest_features = chi2_feature.fit_transform(X_cat,y)