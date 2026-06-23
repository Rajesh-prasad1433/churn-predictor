from flask import Flask,request,jsonify
import pickle
import pandas as pd

app=Flask(__name__)

#Load everything model scaler encodes etc
model=pickle.load(open("models/churn-predictor.pkl","rb"))
encoders=pickle.load(open("models/encoders.pkl","rb"))
scaler=pickle.load(open("models/scaler.pkl","rb"))
numeric_col=['tenure','MonthlyCharges','TotalCharges']

@app.route("/predict",methods=["POST"])

def predict():
    data=request.get_json()
    df=pd.DataFrame([data])

    for col,le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    df[numeric_col] = scaler.transform(df[numeric_col])
    prediction=model.predict(df)[0]
    result="Yes" if prediction == 1 else "No"
    return jsonify({"Churn prediction": result})
if __name__ == "__main__":
    app.run(debug=True)
