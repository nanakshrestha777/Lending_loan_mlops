# import pandas as pd
# import numpy as np
# from sqlalchemy import create_engine
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import joblib
# import mlflow
# import mlflow.sklearn
# import os


# CONN_STR = "mysql+pymysql://admin:admin@127.0.0.1:3306/lending_club"
# engine = create_engine(CONN_STR)


# os.makedirs("../models", exist_ok=True)


# mlflow.set_experiment("Lending_Club_Credit_Risk")



# # --- 2. LOAD DATA ---


# df = pd.read_sql("SELECT * FROM loans_cleaned", engine)



# X = df.drop('target', axis=1)
# y = df['target']


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





# with mlflow.start_run():
    

#     params = {"solver": "saga", "max_iter": 200, "C": 1.0}
#     mlflow.log_params(params)
    
#     # B. Fit the Model
#     model = LogisticRegression(**params)
#     model.fit(X_train, y_train)
    
#     # C. Evaluate

#     predictions = model.predict(X_test)
#     accuracy = accuracy_score(y_test, predictions)
    
#     print(f"   -> Model Accuracy: {accuracy:.4f}")
#     print("   -> Classification Report:\n", classification_report(y_test, predictions))
    
#     # D. Log Metrics to MLflow
#     mlflow.log_metric("accuracy", accuracy)
    
#     # E. Save Model for Deployment
#     # 1. Save to MLflow
#     mlflow.sklearn.log_model(model, "logistic_regression_model")
    
#     # 2. Save locally (for the API in Step 4)
#     joblib.dump(model, "../models/loan_model.pkl")


# print("Model is trained and ready.")
