import pandas as pd
import numpy as np
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Load data
mtrain = pd.read_csv("maritime_train (1).csv")

y = mtrain['Outcome']
mtrain.drop(['PassengerId','PassengerName','Berth','Outcome'], axis=1, inplace=True)
x = mtrain.copy()

# Age mean
age_mean = x['Age'].mean()
x['Age'] = x['Age'].fillna(age_mean)

# Ticket prefix
def get_prefix(ticket):
    ticket = ticket.strip()
    prefix = re.sub(r'[^A-Za-z/]', '', ticket)
    return 'Numeric' if prefix == '' else prefix.upper()

x['TicketPrefix'] = x['CLass'].apply(get_prefix)
x.drop('CLass', axis=1, inplace=True)

# Preprocessing
num_cols = ['TicketTier','Age','RelativesAboard','ParentsChildren',
            'TicketCost','FamilySize','FarePerPerson']
cat_ohe = ['Gender','BoardingPort','Title','TicketPrefix']

num_t = Pipeline([('scaler', StandardScaler())])
cat_t = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])
pp = ColumnTransformer([('num', num_t, num_cols), ('cat', cat_t, cat_ohe)])

x_pp = pp.fit_transform(x)

# Train
model = LogisticRegression(penalty='l1', solver='saga', max_iter=1000, random_state=42)
model.fit(x_pp, y)

# Save artifacts
joblib.dump(model, "model.pkl")
joblib.dump(pp, "preprocessor.pkl")
joblib.dump(age_mean, "age_mean.pkl")

print("Done! Model saved.")