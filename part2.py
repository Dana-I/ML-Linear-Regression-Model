from ucimlrepo import fetch_ucirepo
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#scikit learn package ---> SGDRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

def log_results_SGD(filename, eta0, max_iter, train_mse, test_mse, r2, ev):
    with open(filename, "a") as f:
        f.write(f"eta0: {eta0}\n")
        f.write(f"max_iter: {max_iter}\n")
        f.write(f"Training MSE: {train_mse}\n")
        f.write(f"Test MSE: {test_mse}\n")
        f.write(f"R2 Score: {r2}\n")
        f.write(f"Explained Variance: {ev}\n")
        f.write("-" * 40 + "\n")


#parameter tuning 
# with open("training_log_part2.txt", "w") as f: 
#     f.write("Part 2 SGDRegressor Tuning Log\n")
#     f.write('=' * 40 + "\n")

# eta0_list = [0.01, 0.001]
# max_iter_list = [2000,5000, 10000]

# best_eta0 = None
# best_max_iter = None
# best_test_mse = float("inf")

# for eta0 in eta0_list:
#     for max_iter in max_iter_list:
#         model = SGDRegressor(
#             learning_rate = "constant",
#             eta0 = eta0,
#             max_iter = max_iter,
#             alpha = 0.0001,
#             penalty = "l2",
#             tol = 1e-4,
#             random_state = 42
#         )

#         model.fit(X_train, y_train)

#         train_pred = model.predict(X_train)
#         test_pred = model.predict(X_test)

#         train_mse = mean_squared_error(y_train, train_pred)
#         test_mse = mean_squared_error(y_test, test_pred)
#         r2 = r2_score(y_test, test_pred)
#         ev = explained_variance_score(y_test, test_pred)

#         print(f"eta0={eta0}, max_iter={max_iter}")
#         print("Train MSE:", train_mse)
#         print("Test MSE:", test_mse)
#         print("R2:", r2)
#         print("-" * 30)

#         log_results_SGD("training_log_part2.txt", eta0, max_iter, train_mse, test_mse, r2, ev)

#         if test_mse < best_test_mse:
#             best_test_mse = test_mse
#             best_eta0 = eta0
#             best_max_iter = max_iter

# print("Best eta0:", best_eta0)
# print("Best max_iter:", best_max_iter)
# print("Best Test MSE:", best_test_mse)

best_model = SGDRegressor(
    learning_rate="constant",
    eta0=0.001,
    max_iter=2000,
    alpha=0.0001,
    penalty="l2",
    tol=1e-4,
    random_state=42
)

best_model.fit(X_train, y_train)

train_pred = best_model.predict(X_train)
test_pred = best_model.predict(X_test)

train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)
test_ev = explained_variance_score(y_test, test_pred)

print("\nFinal SGDRegressor Results (Best Params)")
print("Training MSE:", train_mse)
print("Test MSE:", test_mse)
print("Test R2:", test_r2)
print("Explained Variance:", test_ev)
print("Intercept:", best_model.intercept_)
print("Weights:", best_model.coef_)

#plots
plt.scatter(y_test, test_pred, alpha=0.7)
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("SGDRegressor (Best Params): Predicted vs Actual Final Grades")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], "r--")
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(feature_names, best_model.coef_)
plt.xticks(rotation=90)
plt.ylabel("Weight Value")
plt.title("SGDRegressor Feature Weights")
plt.tight_layout()
plt.show()

plt.scatter(student_performance.data.features["studytime"], y, alpha=0.7)
plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")
plt.title("G3 vs Study Time")
plt.show()
