import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
import os
os.chdir(os.path.dirname (os.path.realpath (__file__)))
def model_train(X, y, model, display=False):
    """
    Train the model with the provided features and target variable.
    
    Parameters:
    X (pd.DataFrame): Features for training.
    y (pd.Series): Target variable for training.
    model: The machine learning model to train.
    
    Returns:
    model: The trained model.
    """
    model.fit(X, y)
    loo = LeaveOneOut()
    mse_scores = -cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
    if display:
        print(f"LOOCV MSE: {mse_scores.mean():.4f} ± {mse_scores.std():.4f}")
    mse = mse_scores.mean()
    return model, mse

def model_validate(X, y, model):
    """
    Validate the model with the provided features and target variable.
    
    Parameters:
    X (pd.DataFrame): Features for validation.
    y (pd.Series): Target variable for validation.
    model: The machine learning model to validate.
    
    Returns:
    metrics: Dictionary containing R^2 and MSE of the validation.
    """
    predictions = model.predict(X)
    r2 = r2_score(y, predictions)
    mse = mean_squared_error(y, predictions)
    print(f"Model validated with R^2: {r2:.4f}, MSE: {mse:.4f}")
    metrics = {'r2': r2, 'mse': mse}
    return predictions, metrics

def plot_predictions(y_true, y_pred, title='Model Predictions'):
    """
    Plot the true vs predicted values.
    
    Parameters:
    y_true (pd.Series): True target values.
    y_pred (np.ndarray): Predicted values from the model.
    title (str): Title of the plot.
    """
    plt.figure(figsize=(3, 3))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(title)
    minimum = min(y_true.min(), y_pred.min())
    maximum = max(y_true.max(), y_pred.max())
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r', lw=2)
    plt.xlim(0.9*minimum, 1.1*maximum)
    plt.ylim(0.9*minimum, 1.1*maximum)
    plt.grid()
    plt.show()

def read_data(file_path):
    """
    Read data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: DataFrame containing the data.
    """
    os.chdir(r'data')
    back = os.path.dirname(os.getcwd())
    data = pd.read_excel(file_path)
    x = np.array(data.drop(columns=['con.', 'BD']))
    y_conversion = np.array(data['con.'])
    y_BD = np.array(data['BD'])
    os.chdir(back)
    return x, y_conversion, y_BD

def data_process(x):
    df = pd.DataFrame(x)
    df[5] = (1.06e20*np.exp(-1.273e4/(x[:,1]+273.15))+1.21e21*np.exp(-1.324e4/(x[:,1]+273.15)))/(2.97e21*np.exp(-1.293e4/(x[:,1]+273.15)) + 1.06e20*np.exp(-1.273e4/(x[:,1]+273.15))+1.21e21*np.exp(-1.324e4/(x[:,1]+273.15)))
    return np.array(df)
# Import and process data
from sklearn.preprocessing import StandardScaler

train_x, train_y_con, train_y_BD = read_data('train_BD.xlsx')
train_x = data_process(train_x)
test_x, test_y_con, test_y_BD = read_data('test_BD.xlsx')
test_x = data_process(test_x)
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

# Models
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
Linear =  LinearRegression(fit_intercept=True)
Ridge = model = Ridge(alpha=1)
Quadratic = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
SVM = SVR(kernel='rbf', C=1.0, epsilon=0.1)
kNN = KNeighborsRegressor(n_neighbors=3, weights='distance')
XGBoost = XGBRegressor(n_estimators=20, max_depth=2, learning_rate=0.1)
Neural = MLPRegressor(
    hidden_layer_sizes=(3,), 
    activation='relu',
    solver='lbfgs',          
    alpha=0.5,             
    max_iter=300,
    random_state=42
)
RandomForest = RandomForestRegressor(n_estimators=10, max_depth=2, random_state=42)

# Model Training
trained_models = {}
model_names = ['Linear', 'Ridge', 'Quadratic', 'SVM', 'kNN', 'XGBoost', 'Neural', 'RandomForest']
models = [Linear, Ridge, Quadratic, SVM, kNN, XGBoost, Neural, RandomForest]
for model, name in zip(models, model_names):
    print(f"Training {name} model...")
    trained_model, mse = model_train(train_x_scaled, train_y_BD, model)
    trained_models[name] = trained_model
    print(f"{name} model trained with MSE: {mse:.4f}")

# Plot
for name, model in trained_models.items():
    print(f"Validating {name} model...")
    predictions, metrics = model_validate(test_x_scaled, test_y_BD, model)
    plot_predictions(test_y_BD, predictions, title=f'{name} Model Predictions')
    print(f"{name} model validation metrics: {metrics}")
    print("=" * 50)

import pickle as pkl
def save_model(model, filename):
    """
    Save the trained model to a file.
    
    Parameters:
    model: The trained model to save.
    filename (str): The name of the file to save the model to.
    """
    with open(filename, 'wb') as f:
        pkl.dump(model, f)
    print(f"Model saved to {filename}")

# Save Model
import os
train_x, train_y_con, train_y_BD = read_data('BD.xlsx')
train_x = data_process(train_x)
train_x_scaled = scaler.fit_transform(train_x)
model_names = ['Linear']
models = [Linear]
os.chdir(r'model')
for model, name in zip(models, model_names):
    print(f"Training {name} model...")
    trained_model, mse = model_train(train_x_scaled, train_y_BD, model)
    trained_models[name] = trained_model
    print(f"{name} model trained with MSE: {mse:.4f}")
save_model(trained_models['Linear'], 'BD.pkl')
save_model(scaler, 'scaler_BD.pkl')