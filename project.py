# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
file_path = r"/mnt/c/Users/saran/Desktop/movies/dataset.csv"
data = pd.read_csv(file_path)

scaler = StandardScaler()
num_features = ['Signal_Strength', 'Latency', 'Required_Bandwidth']
data[num_features] = scaler.fit_transform(data[num_features])
# %%

if 'Application_Type' in data.columns:
    encoder = OneHotEncoder(sparse_output=False)
    applications_encoded = encoder.fit_transform(data[['Application_Type']])
    applications_df = pd.DataFrame(applications_encoded, columns=encoder.get_feature_names_out(['Application_Type']))
    data = pd.concat([data, applications_df], axis=1).drop('Application_Type', axis=1)
else:
    print("Column 'Application_Type' is not found in the dataset. Skipping one-hot encoding.")

X = data.drop(['Allocated_Bandwidth'], axis=1)
y = data['Allocated_Bandwidth']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.activations import gelu
# %%

model = Sequential([
    Dense(256, activation=gelu, input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(128, activation=gelu),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.summary()
def convert_bandwidth(value):
    if isinstance(value, float):
        return value
    elif 'Kbps' in value:
        return float(value.replace(' Kbps', '')) / 1000
    elif 'Mbps' in value:
        return float(value.replace(' Mbps', ''))
    else:
        raise ValueError(f"Unexpected unit in value: {value}")

if 'Required_Bandwidth' in data.columns:
    data['Required_Bandwidth'] = data['Required_Bandwidth'].apply(convert_bandwidth)
else:
    print("Column 'Required_Bandwidth' is not in the dataset.")

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)
y_train = pd.to_numeric(y_train, errors='coerce')
y_test = pd.to_numeric(y_test, errors='coerce')
y_train.fillna(0, inplace=True)
y_test.fillna(0, inplace=True)
from tensorflow.keras.callbacks import EarlyStopping
# %%

early_stopping = EarlyStopping(
    monitor='val_loss',  
    patience=10,         
    restore_best_weights=True  
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=500,         
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

print(f"Stopped at epoch: {len(history.history['loss'])}")

loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss (MSE): {loss}")
print(f"Test Mean Absolute Error (MAE): {mae}")
predictions = model.predict(X_test)

print("Sample Predictions:")
for actual, predicted in zip(y_test[:10], predictions[:10]):
    print(f"Actual: {actual}, Predicted: {predicted[0]:.2f}")
# %%

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Values')
plt.scatter(range(len(predictions)), predictions, color='red', label='Predicted Values')
plt.title('Actual vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Allocated Bandwidth') 
plt.legend()
plt.show()
# %%

#random forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
mse = mean_squared_error(y_test, rf_predictions)
mae = mean_absolute_error(y_test, rf_predictions)

print(f"Random Forest Test MSE: {mse:.2f}")
print(f"Random Forest Test MAE: {mae:.2f}")
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [10, 20, None],     # Maximum depth of each tree
    'min_samples_split': [2, 5, 10], # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],   # Minimum samples required at a leaf node
    'max_features': ['sqrt', 'log2', None]  # Number of features to consider when looking for the best split
}
rf_model = RandomForestRegressor(random_state=42)
# %%

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='neg_mean_absolute_error',
    cv=3,
    verbose=2,
    n_jobs=-1,
    error_score='raise'
)

grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
best_rf_model = RandomForestRegressor(
    max_depth=10,
    max_features='sqrt',
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100,
    random_state=42
)

best_rf_model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error, mean_absolute_error

rf_predictions = best_rf_model.predict(X_test)

mse = mean_squared_error(y_test, rf_predictions)
mae = mean_absolute_error(y_test, rf_predictions)

print(f"Test MSE: {mse:.2f}")
print(f"Test MAE: {mae:.2f}")
import matplotlib.pyplot as plt

feature_importances = best_rf_model.feature_importances_
# %%

plt.figure(figsize=(10, 6))
plt.barh(X_train.columns, feature_importances, color='skyblue')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance in Random Forest')
plt.show()
# %%

#SVM Regression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

svm_model = SVR(kernel='rbf', C=100, gamma=0.1)
svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)
svm_mse = mean_squared_error(y_test, svm_predictions)
svm_mae = mean_absolute_error(y_test, svm_predictions)

print("SVM Results:")
print(f"Mean Squared Error (MSE): {svm_mse:.2f}")
print(f"Mean Absolute Error (MAE): {svm_mae:.2f}")

# Plot predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Values')
plt.scatter(range(len(svm_predictions)), svm_predictions, color='red', label='Predicted Values', alpha=0.6)
plt.title('SVM: Predicted vs Actual Values')
plt.xlabel('Sample Index')
plt.ylabel('Bandwidth')
plt.ylim(0, 40)
plt.legend()
plt.show()
# %%

#GBT Regression
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

gbt_model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
gbt_model.fit(X_train, y_train)

gbt_predictions = gbt_model.predict(X_test)
gbt_mse = mean_squared_error(y_test, gbt_predictions)
gbt_mae = mean_absolute_error(y_test, gbt_predictions)

print("Gradient Boosted Trees Results:")
print(f"Mean Squared Error (MSE): {gbt_mse:.2f}")
print(f"Mean Absolute Error (MAE): {gbt_mae:.2f}")

# Plot predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Values')
plt.scatter(range(len(gbt_predictions)), gbt_predictions, color='red', label='Predicted Values', alpha=0.6)
plt.title('GBT: Predicted vs Actual Values')
plt.xlabel('Sample Index')
plt.ylabel('Bandwidth')
plt.ylim(0, 40)
plt.legend()
plt.show()
# %%

#KNN regression
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt

knn_model = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto')
knn_model.fit(X_train, y_train)

knn_predictions = knn_model.predict(X_test)
knn_mse = mean_squared_error(y_test, knn_predictions)
knn_mae = mean_absolute_error(y_test, knn_predictions)

print("KNN Results:")
print(f"Mean Squared Error (MSE): {knn_mse:.2f}")
print(f"Mean Absolute Error (MAE): {knn_mae:.2f}")

# Plot predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Values')
plt.scatter(range(len(knn_predictions)), knn_predictions, color='red', label='Predicted Values', alpha=0.6)
plt.title('KNN: Predicted vs Actual Values')
plt.xlabel('Sample Index')
plt.ylabel('Bandwidth')
plt.ylim(0, 40)
plt.legend()
plt.show()
# %%

#Comparison of different models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

results = {}

# Function to evaluate and store metrics for each model
def evaluate_model(model_name, y_test, predictions):
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    results[model_name] = {'MSE': mse, 'MAE': mae, 'R2': r2}

# Evaluate SVM
svm_model = SVR(kernel='rbf', C=100, gamma=0.1)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
evaluate_model('SVM', y_test, svm_predictions)

# Evaluate Gradient Boosted Trees
from sklearn.ensemble import GradientBoostingRegressor
gbt_model = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42)
gbt_model.fit(X_train, y_train)
gbt_predictions = gbt_model.predict(X_test)
evaluate_model('GBT', y_test, gbt_predictions)

# Evaluate KNN
from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto')
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
evaluate_model('KNN', y_test, knn_predictions)

# Evaluate Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
evaluate_model('Random Forest', y_test, rf_predictions)

# Determine the best model based on Mean Squared Error (lower is better)
best_model_name = min(results, key=lambda model: results[model]['MSE'])
best_model_metrics = results[best_model_name]

print(f"Best Model: {best_model_name}")
print(f"MSE: {best_model_metrics['MSE']:.2f}, MAE: {best_model_metrics['MAE']:.2f}, R²: {best_model_metrics['R2']:.2f}")

# Convert results into a plottable format
model_names = list(results.keys())
mse_values = [results[model]['MSE'] for model in model_names]
mae_values = [results[model]['MAE'] for model in model_names]
r2_values = [results[model]['R2'] for model in model_names]

# Visualization of all models
plt.figure(figsize=(16, 6))

# Plot MSE
plt.subplot(1, 3, 1)
plt.bar(model_names, mse_values, color='skyblue')
plt.title('Mean Squared Error (MSE)')
plt.ylabel('Error')

# Plot MAE
plt.subplot(1, 3, 2)
plt.bar(model_names, mae_values, color='orange')
plt.title('Mean Absolute Error (MAE)')
plt.ylabel('Error')

# Plot R² Scores
plt.subplot(1, 3, 3)
plt.bar(model_names, r2_values, color='green')
plt.title('R² Score')
plt.ylabel('Score')

plt.tight_layout()
plt.show()

# Highlight the Best Model
print(f"\nThe best-performing model is **{best_model_name}** with the following metrics:")
print(f" - Mean Squared Error (MSE): {best_model_metrics['MSE']:.2f}")
print(f" - Mean Absolute Error (MAE): {best_model_metrics['MAE']:.2f}")
print(f" - R² Score: {best_model_metrics['R2']:.2f}")

#%%
# Import required libraries
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import individual models and test datasets from project.py
from project import knn_model, rf_model, gbt_model, svm_model, X_test, y_test

# Initialize the CustomTkinter window
ctk.set_appearance_mode("System")  # Light, Dark, or System theme
ctk.set_default_color_theme("blue")  # Default blue theme
root = ctk.CTk()
root.title("Model Performance Dashboard")
root.geometry("1000x700")

# Prepare models and their names
models = {
    "KNN Model": knn_model,
    "Random Forest Model": rf_model,
    "Gradient Boosting Model": gbt_model,
    "SVM Model": svm_model
}

# Function to calculate model performance metrics
def calculate_metrics():
    metrics = {}
    for model_name, model in models.items():
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        metrics[model_name] = {"MSE": mse, "MAE": mae}
    return metrics

# Pre-calculate metrics for all models
results = calculate_metrics()

# Function to display performance metrics and plot the selected model's data
def show_metrics():
    selected_model = model_choice.get()
    metrics_label.configure(
        text=f"Model: {selected_model}\n"
             f"Mean Squared Error (MSE): {results[selected_model]['MSE']:.2f}\n"
             f"Mean Absolute Error (MAE): {results[selected_model]['MAE']:.2f}"
    )
    plot_chart(selected_model)

# Function to generate a bar chart for the selected model
def plot_chart(model_name):
    mse = results[model_name]["MSE"]
    mae = results[model_name]["MAE"]

    # Generate bar chart
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(["MSE", "MAE"], [mse, mae], color=["#1f77b4", "#ff7f0e"])
    ax.set_title(f"{model_name} Metrics")
    ax.set_ylabel("Error")
    ax.set_xlabel("Metrics")

    # Clear previous chart and display the updated one
    for widget in chart_frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)

# Dropdown menu for selecting a model
model_label = ctk.CTkLabel(root, text="Select a Model:", font=("Helvetica", 16))
model_label.pack(pady=10)
model_choice = ctk.CTkComboBox(root, values=list(models.keys()), font=("Helvetica", 14))
model_choice.pack(pady=5)
model_choice.set(list(models.keys())[0])  # Set default selection

# Button to show metrics and chart for the selected model
show_button = ctk.CTkButton(root, text="Show Metrics", command=show_metrics, font=("Helvetica", 14))
show_button.pack(pady=10)

# Label to display model metrics
metrics_label = ctk.CTkLabel(root, text="", font=("Helvetica", 14))
metrics_label.pack(pady=10)

# Frame for displaying charts
chart_frame = ctk.CTkFrame(root, width=800, height=500)
chart_frame.pack(pady=20)

# Run the GUI application
root.mainloop()
# %%
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"/mnt/c/Users/saran/Desktop/movies/dataset.csv")

# Ensure 'Timestamp' column is in datetime format
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%H:%M:%S')

# Convert the 'Timestamp' column to string format (HH:MM:SS)
data['Timestamp'] = data['Timestamp'].dt.strftime('%H:%M:%S')

# Generate predicted values using the models
# Assuming `rf_predictions`, `svm_predictions`, `knn_predictions`, and `gbt_predictions` are already computed
rf_predictions = rf_model.predict(X_test)
svm_predictions = svm_model.predict(X_test)
knn_predictions = knn_model.predict(X_test)
gbt_predictions = gbt_model.predict(X_test)

# Plot the actual Required_Bandwidth values
plt.figure(figsize=(12, 6))
plt.scatter(data['Timestamp'], data['Required_Bandwidth'], color='blue', marker='o', label="Actual Required Bandwidth")

# Overlay predicted values for each model
plt.scatter(data['Timestamp'][:len(rf_predictions)], rf_predictions, color='red', marker='x', label="RF Predicted Bandwidth")
plt.scatter(data['Timestamp'][:len(svm_predictions)], svm_predictions, color='green', marker='v', label="SVM Predicted Bandwidth")
plt.scatter(data['Timestamp'][:len(knn_predictions)], knn_predictions, color='orange', marker='s', label="KNN Predicted Bandwidth")
plt.scatter(data['Timestamp'][:len(gbt_predictions)], gbt_predictions, color='purple', marker='d', label="GBT Predicted Bandwidth")

# Customize the plot
plt.title("Actual vs Predicted Bandwidth Over Time", fontsize=16)
plt.xlabel("Time (HH:MM:SS)", fontsize=14)
plt.ylabel("Required Bandwidth (units)", fontsize=14)
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend(loc="upper left")

# Show the plot
plt.tight_layout()  # Adjust layout to avoid clipping labels
plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv(r"/mnt/c/Users/saran/Desktop/movies/dataset.csv")

# Ensure 'Timestamp' column is in datetime format
data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='%H:%M:%S')

# Convert the 'Timestamp' column to string format (HH:MM:SS)
data['Timestamp'] = data['Timestamp'].dt.strftime('%H:%M:%S')

# Generate predicted values using the models
rf_predictions = rf_model.predict(X_test)
svm_predictions = svm_model.predict(X_test)
knn_predictions = knn_model.predict(X_test)
gbt_predictions = gbt_model.predict(X_test)

# Define a function to create separate plots
def plot_actual_vs_predicted(timestamp, actual, predicted, model_name):
    plt.figure(figsize=(12, 6))
    # Plot actual bandwidth
    plt.scatter(timestamp[:len(actual)], actual[:len(timestamp)], color='blue', marker='o', label="Actual Required Bandwidth")
    # Plot predicted bandwidth for the model
    plt.scatter(timestamp[:len(predicted)], predicted[:len(timestamp)], color='red', marker='x', label=f"Predicted Bandwidth ({model_name})")

    # Customize the plot
    plt.title(f"Actual vs Predicted Bandwidth ({model_name})", fontsize=16)
    plt.xlabel("Time (HH:MM:SS)", fontsize=14)
    plt.ylabel("Required Bandwidth (units)", fontsize=14)
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.legend(loc="upper left")
    plt.tight_layout()  # Adjust layout to avoid clipping labels
    plt.show()

# Plot for Random Forest Model
plot_actual_vs_predicted(data['Timestamp'], data['Required_Bandwidth'], rf_predictions, "Random Forest Model")

# Plot for SVM Model
plot_actual_vs_predicted(data['Timestamp'], data['Required_Bandwidth'], svm_predictions, "SVM Model")

# Plot for KNN Model
plot_actual_vs_predicted(data['Timestamp'], data['Required_Bandwidth'], knn_predictions, "KNN Model")

# Plot for Gradient Boosting Model
plot_actual_vs_predicted(data['Timestamp'], data['Required_Bandwidth'], gbt_predictions, "Gradient Boosting Model")

# %%
