import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics 
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import pickle
from sklearn.ensemble import VotingClassifier


#Reading the data
data=pd.read_csv("diabetes_binary_health_indicators_BRFSS2015.csv")

#Removing duplicates
data.drop_duplicates(keep='first' ,inplace=True)

# split into input (x) and output (y) variables
x=data.drop(columns={'Diabetes_binary','Education','NoDocbcCost'},axis=1)
y=data['Diabetes_binary']

#detect outliers and removing them
ax = data[['BMI','PhysHlth','GenHlth','MentHlth']].plot(kind='box', title='outliers',figsize=(10,10))
plt.show()
out=RobustScaler().fit(x)
x=out.transform(x)

#visualize correlation 
corr=data.corr().round(2)
sns.heatmap(corr)
plt.show()

#oversampling
ros= RandomOverSampler(sampling_strategy="minority")
x_new,y_new=ros.fit_resample(x,y)
y_new.value_counts().plot.pie(autopct='%.2f')

#Standardizing the features
scaler=StandardScaler()
scaler.fit(x_new)
standardX=scaler.transform(x_new)
x_new=standardX

#splitting the dataset
x_train,x_test, y_train, y_test=train_test_split(x_new,y_new,test_size=0.30,shuffle=True)


#creating svm model and training it
model_svm= SVC(kernel ='sigmoid',max_iter=(5000))
model_svm.fit(x_train,y_train)
y_pred_svm=model_svm.predict(x_test)
accuracy_svm=metrics.accuracy_score(y_test,y_pred_svm)
print('\n\n ')
print('----------------------------------------------------')
print(' SVM model ')
print('accuracy in percentage of svm model: ' , int(accuracy_svm*100),'%')
print('confusion matrix:\n ' , metrics.confusion_matrix(y_test, y_pred_svm))
print('Classification report:\n ', metrics.classification_report(y_test, y_pred_svm))
print('----------------------------------------------------')

#creating logistic model and training it
model_logistic=LogisticRegression()
model_logistic.fit(x_train,y_train)
predict_logistic=model_logistic.predict_proba(x_test)
y_pred_logistic=model_logistic.predict(x_test)
accuracy_logistic=metrics.accuracy_score(y_test,y_pred_logistic)
print(' Logistic model ')
print('accuracy in percentage of Logistic model: ' , int(accuracy_logistic*100),'%')
print('confusion matrix:\n ' , metrics.confusion_matrix(y_test, y_pred_logistic))
print('Classification report:\n ', metrics.classification_report(y_test, y_pred_logistic))
print('----------------------------------------------------')

#creating Decision Tree model and training it
model_tree=DecisionTreeClassifier()
model_tree.fit(x_train,y_train)
y_pred_tree=model_tree.predict(x_test)
accuracy_tree=metrics.accuracy_score(y_test,y_pred_tree)
print(' Tree model ')
print('accuracy in percentage of decision tree model: ' , int(accuracy_tree*100),'%')
print('confusion matrix:\n ' , metrics.confusion_matrix(y_test, y_pred_tree))
print('Classification report:\n ', metrics.classification_report(y_test, y_pred_tree))
print('----------------------------------------------------')

#Combining models
combined = VotingClassifier(
    estimators = [('dtc',DecisionTreeClassifier()),
                  ('lr', LogisticRegression()),
                  ('svm', SVC(kernel ='sigmoid',max_iter=(5000)))],voting='hard')
combined.fit(x_train, y_train)
y_pred_vch = combined.predict(x_test)
accuracy_vch=metrics.accuracy_score(y_test, y_pred_vch)
print('accuracy in percentage of combined models: ' , int(accuracy_vch*100),'%')
print('confusion matrix:\n ' , metrics.confusion_matrix(y_test, y_pred_vch))
print('Classification report:\n ', metrics.classification_report(y_test, y_pred_vch))
print('----------------------------------------------------')

#testing logistic model
input_data = (1,1,1,30,1,0,1,0,1,1,0,1,5,30,30,1,0,9,1)
inputdata_np = np.asarray(input_data)
input_data_reshaped = inputdata_np.reshape(1,-1)
newdata=out.transform(input_data_reshaped)
newdata=scaler.transform(newdata)
prediction = combined.predict(newdata)
print(prediction)
print('----------------------------------------------------')
print('Combined testing')

if (prediction == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
print('----------------------------------------------------')

#testing tree model
input_data=(1,1,1,28,0,0,0,0,1,0,0,1,5,30,30,1,0,9,8)
inputdata_np = np.asarray(input_data)
input_data_reshaped = inputdata_np.reshape(1,-1)
newdata=out.transform(input_data_reshaped)
newdata=scaler.transform(newdata)
prediction = model_tree.predict(newdata)
print(prediction)
print('----------------------------------------------------')
print('Desicion tree testing')

if (prediction == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
print('----------------------------------------------------')

#saving models
svm='model_svm.sav'
pickle.dump(model_svm,open(svm,'wb'))

logistic='model_logistic.sav'
pickle.dump(model_logistic,open(logistic,'wb'))

tree='model_tree.sav'
pickle.dump(model_tree,open(tree,'wb'))

combined_model='combined.sav'
pickle.dump(combined,open(combined_model,'wb'))

"""#loading the model and using it
model_loaded = pickle.load(open('model_tree.sav', 'rb'))
y_pred_model=model_loaded.predict(x_test)
print('Score: ',metrics.accuracy_score(y_test, y_pred_model))
print('Classification report:\n ', metrics.classification_report(y_test,y_pred_model))"""
