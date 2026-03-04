import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


df = pd.read_csv("traffic_training_data.csv")


print("\nClass Distribution:")
print(df["congestion"].value_counts())

X = df[["vehicle_count", "avg_count", "hour"]]
y = df["congestion"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)


predictions = model.predict(X_test)

print("\nModel Evaluation:\n")
print(classification_report(y_test, predictions))

joblib.dump(model, "congestion_model_new.pkl")

print("\nNew model saved as congestion_model.pkl")