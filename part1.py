from ucimlrepo import fetch_ucirepo
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


student_performance = fetch_ucirepo(id=320)
X = student_performance.data.features.copy()
y = student_performance.data.targets.copy()


y = y["G3"]
X = X.drop(columns=["G1", "G2"], errors="ignore")

X = pd.get_dummies(X, drop_first = True)
feature_names = list(X.columns)

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

class LinearRegressionGD:
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.loss_history = []

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        
        for i in range(self.iterations):
            y_pred = np.dot(X, self.weights)
            error = y_pred - y
            
            gradient = (2/m) * np.dot(X.T, error)
            self.weights -= self.learning_rate * gradient
            
            mse = np.mean(error**2)
            self.loss_history.append(mse)

    def predict(self, X):
        return np.dot(X, self.weights)

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred)**2)

    def r2_score(self, y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true))**2)
        ss_residual = np.sum((y_true - y_pred)**2)
        return 1 - (ss_residual / ss_total)

def log_results(filename, lr, iterations, train_mse, test_mse, r2):
    with open(filename, "a") as f:
        f.write(f"Learning Rate: {lr}\n")
        f.write(f"Iterations: {iterations}\n")
        f.write(f"Training MSE: {train_mse}\n")
        f.write(f"Test MSE: {test_mse}\n")
        f.write(f"R2 Score: {r2}\n")
        f.write("-" * 40 + "\n")

# Parameter Tuning:
# learning_rates = [0.1, 0.01, 0.001]
# iterations_list = [1000, 3000, 5000]

# for lr in learning_rates:
#     for iters in iterations_list:
        
#         model = LinearRegressionGD(learning_rate=lr, iterations=iters)
#         model.fit(X_train, y_train)

#         train_pred = model.predict(X_train)
#         test_pred = model.predict(X_test)

#         train_mse = model.mse(y_train, train_pred)
#         test_mse = model.mse(y_test, test_pred)
#         r2 = model.r2_score(y_test, test_pred)

#         print(f"LR={lr}, Iter={iters}")
#         print("Train MSE:", train_mse)
#         print("Test MSE:", test_mse)
#         print("R2 Score:", r2)
#         print("---------------------")

#         log_results("training_log.txt", lr, iters, train_mse, test_mse, r2)

best_model = LinearRegressionGD(learning_rate=0.001, iterations=5000)
best_model.fit(X_train, y_train)

train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)

print("Best Model Results")
print("Training MSE:", best_model.mse(y_train, train_pred))
print("Test MSE:", best_model.mse(y_test, test_pred))
print("R2:", best_model.r2_score(y_test, test_pred))
print("Weights:", best_model.weights)

# Plots
plt.plot(best_model.loss_history)
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title("Training MSE vs Iteration")
plt.show()

plt.scatter(y_test, test_pred, alpha=0.7)
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Predicted vs Actual Final Grades")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "r--")  # 45° line
plt.show()

features = ["Intercept"] + feature_names  # include intercept
plt.figure(figsize=(10,5))
plt.bar(features, best_model.weights)
plt.xticks(rotation=90)
plt.ylabel("Weight Value")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

plt.scatter(student_performance.data.features["studytime"], y, alpha=0.7)
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")
plt.title("G3 vs Study Time")
plt.show()
