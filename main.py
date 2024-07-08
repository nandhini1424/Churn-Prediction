import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('Churn.csv')

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

le = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    if column != 'customerID':
        df[column] = le.fit_transform(df[column])

X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

def get_user_input():
    gender = input("Enter gender (Male/Female): ")
    senior_citizen = int(input("Is the customer a senior citizen? (0 for No, 1 for Yes): "))
    partner = input("Does the customer have a partner? (Yes/No): ")
    dependents = input("Does the customer have dependents? (Yes/No): ")
    tenure = int(input("Enter tenure (in months): "))
    phone_service = input("Does the customer have phone service? (Yes/No): ")
    multiple_lines = input("Does the customer have multiple lines? (Yes/No/No phone service): ")
    internet_service = input("Enter internet service type (DSL/Fiber optic/No): ")
    online_security = input("Does the customer have online security? (Yes/No/No internet service): ")
    online_backup = input("Does the customer have online backup? (Yes/No/No internet service): ")
    device_protection = input("Does the customer have device protection? (Yes/No/No internet service): ")
    tech_support = input("Does the customer have tech support? (Yes/No/No internet service): ")
    streaming_tv = input("Does the customer have streaming TV? (Yes/No/No internet service): ")
    streaming_movies = input("Does the customer have streaming movies? (Yes/No/No internet service): ")
    contract = input("Enter contract type (Month-to-month/One year/Two year): ")
    paperless_billing = input("Does the customer use paperless billing? (Yes/No): ")
    payment_method = input("Enter payment method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)): ")
    monthly_charges = float(input("Enter monthly charges: "))
    total_charges = float(input("Enter total charges: "))

    user_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    
    user_df = pd.DataFrame(user_data, index=[0])
    for column in user_df.select_dtypes(include=['object']).columns:
        user_df[column] = le.fit_transform(user_df[column])
    
    return user_df

user_df = get_user_input()
prediction = model.predict(user_df)
prediction_proba = model.predict_proba(user_df)

print(f"Prediction: {'Churn' if prediction[0] == 1 else 'Not Churn'}")
print(f"Probability of Churning: {prediction_proba[0][1]}")
