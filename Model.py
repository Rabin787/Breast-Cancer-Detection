import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


breast_cancer_data = pd.read_csv('breast-cancer.csv')
breast_cancer_data.drop(columns=['id'],inplace=True)

#Encoding the target variable with LabelEncoder
label_encoder=LabelEncoder()
breast_cancer_data['diagnosis']=label_encoder.fit_transform(breast_cancer_data['diagnosis'])

#splitting the data into features and target variable
X_bc=breast_cancer_data.drop(columns=['diagnosis'])
y_bc=breast_cancer_data['diagnosis']

#splitting into training and testing sets
X_bc_train,X_bc_test,y_bc_train,y_bc_test=train_test_split(X_bc,y_bc,test_size=0.2,random_state=42,stratify=y_bc)

#Standarizing the data
scalar_bc=StandardScaler()
X_bc_train=scalar_bc.fit_transform(X_bc_train)
X_bc_test=scalar_bc.transform(X_bc_test)

#Training the model
knn_bc_model=KNeighborsClassifier(n_neighbors=5,metric='euclidean')
knn_bc_model.fit(X_bc_train,y_bc_train)

#saving the model
joblib.dump(knn_bc_model,'breast_cancer_knn_model.joblib')
joblib.dump(scalar_bc,'breast_cancer_scaler.joblib')
