
# importing libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import uniform
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib 
import streamlit as st


def main():   

    data=pd.read_csv('Churn.csv')
    # Dropping columns which are not necessary for prediction
    data = data.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)
    labels = 'Exited', 'Retained'
    # 1st Attribute - Balance Salary Ratio
    data['BalanceSalaryRatio'] = data.Balance/data.EstimatedSalary
    #  2nd Attribute-Tenure By Age
    data['TenureByAge'] = data.Tenure/(data.Age)
    # 3rd Attribute- Credit Score Given Age
    data['CreditScoreGivenAge'] = data.CreditScore/(data.Age)
    continuous_vars = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary', 'BalanceSalaryRatio',
                    'TenureByAge','CreditScoreGivenAge']
    categorical_vars = ['HasCrCard', 'IsActiveMember','Geography', 'Gender']
    data = data[['Exited'] + continuous_vars + categorical_vars]
    data.loc[data.HasCrCard == 0, 'HasCrCard'] = -1
    data.loc[data.IsActiveMember == 0, 'IsActiveMember'] = -1 
    le = LabelEncoder() 
    data['Gender']= le.fit_transform(data['Gender']) 
    data['Geography']= le.fit_transform(data['Geography']) 
    df = pd.get_dummies(data=data, columns=['Gender','Geography'])
    df.columns
    scaler = MinMaxScaler()
    df[continuous_vars] = scaler.fit_transform(df[continuous_vars])
    for col in df:
        print(f'{col}: {df[col].unique()}')
    X = df.drop('Exited',axis='columns')
    y = df['Exited']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    X_train.shape,y_train.shape,X_test.shape,y_test.shape
    features_label = X_train.columns
    forest = RandomForestClassifier (n_estimators = 1000, random_state = 0, n_jobs = -1)
    forest.fit(X_train, y_train)
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i in range(X.shape[1]):
        print ("%2d) %-*s %f" % (i + 1, 30, features_label[i], importances[indices[i]]))
    
    smote = SMOTE()
    print("check1")
    x1=X_train
    y1=y_train
    st.write(X_train)
    st.write(y_train)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    print("check2")
    y_sm.value_counts()

    X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=15, stratify=y_sm)
    y_train.value_counts()

    # Fit Extreme Gradient boosting classifier
    param_grid = {'max_depth': [5,6,7,8], 'gamma': [0.01,0.001,0.001],'min_child_weight':[1,5,10], 'learning_rate': [0.05,0.1, 0.2, 0.3], 'n_estimators':[5,10,20,100]}
    xgb_grid = GridSearchCV(XGBClassifier(), param_grid, cv=5, refit=True, verbose=0)
    xgb_grid.fit(X_train,y_train)
    sgb_temps=best_model(xgb_grid)

    XGB2 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,colsample_bytree=1, gamma=0.001, learning_rate=0.2, max_delta_step=0,max_depth=8,
                        min_child_weight=1, n_estimators=100,n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,reg_alpha=0, 
                        reg_lambda=1, scale_pos_weight=1, seed=None,  subsample=1)
    final_model=XGB2.fit(X_train,y_train)
    joblib.dump(XGB2, 'best_model.joblib') 
    predict()


def best_model(model):
    print(model.best_score_)    
    print(model.best_params_)
    print(model.best_estimator_)
    return model.best_score_



def predict():
    st.title("Customer Retention Model Prediction")
    st.write("Enter the following details:")
    BalanceSalaryRatio=0
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, step=1)
    age = st.number_input("Age", min_value=18, max_value=100, step=1)
    tenure = st.number_input("Tenure", min_value=0, step=1)
    balance = st.number_input("Balance")
    num_of_products = st.number_input("Number of Products", min_value=1, max_value=10, step=1)
    estimated_salary = st.number_input("Estimated Salary")
    gender = st.selectbox("Gender", ["Male", "Female"])
    
    if gender == "Male":
        gender_encoded = 1
    else:
        gender_encoded = 0
    
    try:
        BalanceSalaryRatio = balance / estimated_salary
        TenureByAge = tenure / age
        CreditScoreGivenAge = credit_score / age
        HasCrCard = 1
        IsActiveMember = 1
    except:
        st.write("Please select non-zero values")    

    # Create the input_data DataFrame
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'EstimatedSalary': [estimated_salary],
        'BalanceSalaryRatio': [BalanceSalaryRatio],
        'TenureByAge': [TenureByAge],
        'CreditScoreGivenAge': [CreditScoreGivenAge],
        'HasCrCard': [HasCrCard],
        'IsActiveMember': [IsActiveMember],
        'Gender': [1 if gender_encoded == 1 else 0]
    })
    if st.button("Predict"):
        reg = joblib.load('Bank_Churn.joblib')
        predictions = reg.predict(input_data)
        print(predictions)
        st.write(predictions)
        
        if predictions == 1:
            st.error("Customer is likely to churn.")
        else:
            st.success("Customer is not likely to churn.")
   

if __name__=="__main__":
    predict()
  




