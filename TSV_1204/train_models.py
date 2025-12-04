import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# Load Data
df = pd.read_csv('tsv_pairs_dataset.csv')

feature_cols = ['Small_up_width', 'Small_depth', 'Small_scallps_length', 'Small_scallps_height', 'Small_angle', 'Large_up_width']
target_cols = ['Large_depth', 'Large_scallps_length', 'Large_scallps_height', 'Large_angle']

X = df[feature_cols]
y = df[target_cols]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling (Important for MLP)
scaler_X = StandardScaler()
# scaler_y = StandardScaler() # Optional, usually not needed for regression targets unless range varies wildly. 
# Here depth is ~150-200, angle ~90, scallops ~0.2-1.0. Ranges vary. 
# Scaling Y might help convergence for MLP.
scaler_y = MinMaxScaler() # Map to 0-1

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

def calculate_accuracy(y_true, y_pred):
    # Accuracy = 1 - MAPE
    # Avoid division by zero if any y_true is 0 (unlikely here)
    mape = mean_absolute_percentage_error(y_true, y_pred, multioutput='raw_values')
    accuracies = 1 - mape
    return np.mean(accuracies) * 100, accuracies

print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.\n")

# --- Decision Tree ---
print("--- Decision Tree ---")
dt = DecisionTreeRegressor(random_state=42, max_depth=10) # Constrain depth slightly to avoid overfitting
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

avg_acc_dt, acc_per_col_dt = calculate_accuracy(y_test, y_pred_dt)
print(f"Average Accuracy: {avg_acc_dt:.2f}%")
print("Accuracy per target:")
for col, acc in zip(target_cols, acc_per_col_dt):
    print(f"  {col}: {acc*100:.2f}%")
print(f"R2 Score: {r2_score(y_test, y_pred_dt, multioutput='uniform_average'):.4f}")

# --- Neural Network ---
print("\n--- Neural Network (MLP) ---")
# MLP requires tuning. 
# Hidden layers: (64, 64) is a good start.
# Solver: 'adam' or 'lbfgs' (lbfgs is good for small datasets)
mlp = MLPRegressor(hidden_layer_sizes=(100, 100), 
                   activation='relu', 
                   solver='lbfgs',  # lbfgs often converges better on small datasets
                   max_iter=5000, 
                   random_state=42)

mlp.fit(X_train_scaled, y_train_scaled)
y_pred_scaled_mlp = mlp.predict(X_test_scaled)
y_pred_mlp = scaler_y.inverse_transform(y_pred_scaled_mlp)

avg_acc_mlp, acc_per_col_mlp = calculate_accuracy(y_test, y_pred_mlp)
print(f"Average Accuracy: {avg_acc_mlp:.2f}%")
print("Accuracy per target:")
for col, acc in zip(target_cols, acc_per_col_mlp):
    print(f"  {col}: {acc*100:.2f}%")
print(f"R2 Score: {r2_score(y_test, y_pred_mlp, multioutput='uniform_average'):.4f}")

# Check if we met the goal
if avg_acc_dt > 90 or avg_acc_mlp > 90:
    print("\nSUCCESS: At least one model achieved > 90% accuracy.")
else:
    print("\nWARNING: Accuracy goal not met. Tuning needed.")








