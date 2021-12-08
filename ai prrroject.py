import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
col_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
# load dataset
pima = pd.read_csv("Crop_recommendation modi2.csv")
pima.head()
feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = pima[feature_cols] # Features
y = pima.label # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="gini",max_depth=10)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=8)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
import pickle
# Save the model
filename = 'model.pkl'
pickle.dump(clf, open(filename, 'wb'))
list_of_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
input_data=pd.DataFrame(columns=list_of_columns)


input_data.at[0,'N']=int(input('enter nitrogen level(N)'))
input_data.at[0,'P']=int(input('enter phosphorus level(P)'))
input_data.at[0,'K']=int(input('enter potassium level(K)'))
input_data.at[0,'temperature']=float(input ('enter temperature'))
input_data.at[0,'humidity']=float(input ('enter humidity '))
input_data.at[0,'ph']=float(input('enter pH value of soul'))
input_data.at[0,'rainfall']=float(input('enter rainfall value'))

model = pickle.load(open('model.pkl', 'rb'))
prediction = model.predict(input_data)
result = prediction[0]
print('crop  ',result)