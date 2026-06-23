import pandas as pd
def loadData(filepath):

    df=pd.read_csv(filepath)

    return df
def cleanData(df):
    df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')

    df=df.dropna(subset=['TotalCharges'])
    return df
if __name__ == "__main__":
    df=loadData("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    print("before cleaning:",df.shape)
    
    df = cleanData(df)

    print("After cleaning:",df.shape)
    print(df.head())


