from ucimlrepo import fetch_ucirepo
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


student_performance = fetch_ucirepo(id=320)
X = student_performance.data.features.copy()
y = student_performance.data.targets.copy()


y = y["G3"]
X = X.drop(columns=["G1", "G2"], errors="ignore")

X = pd.get_dummies(X, drop_first = True)

X = X.to_numpy(dtype = float)
y = y.to_numpy(dtype = float)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = np.c_[np.ones((X.shape[0], 1)), X]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("First 5 y values:", y[:5])
