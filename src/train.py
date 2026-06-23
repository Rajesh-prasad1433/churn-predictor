import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from data_preprocessing import cleanData,loadData
import pickle

def encoding(df):
    df=df.drop('customerID',axis=1)
    cat_col=df.select_dtypes(include=['object','string']).columns
   
    numeric_cols=['tenure','MonthlyCharges','TotalCharges']
    encodes={}
    for col in cat_col:
        le=LabelEncoder()
        df[col]=le.fit_transform(df[col])
        encodes[col]=le
    s=StandardScaler()
    df[numeric_cols]=s.fit_transform(df[numeric_cols])

    return df,encodes,s
def train(df):
    X=df.drop(['Churn'],axis=1)
    Y=df['Churn']
    X_train,x_test,Y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
    model=LogisticRegression(max_iter=1000)
    model.fit(X_train,Y_train)
    predictions=model.predict(x_test)
    accuracy=accuracy_score(y_test,predictions)
    return model,accuracy
if __name__ == "__main__":
    df=loadData("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df=cleanData(df)
    df,encodes,s=encoding(df)
    model,accuracy=train(df)
    print(f"Model accuracy {accuracy:.4f}")

    # Save trained model 
    with open("models/churn-predictor.pkl","wb") as f:
        pickle.dump(model,f)
    with open("models/encoders.pkl","wb") as f:
        pickle.dump(encodes,f)
    print("Saved model and encoders")
    with open("models/scaler.pkl","wb") as f:
        pickle.dump(s,f)


