import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
def data_process_con(x):
    df = pd.DataFrame(x)
    df[5] = 10*x[:,2]*x[:,3]*x[:,4]*np.exp(-1000*(1/(x[:,1]+273.15)-1/(40+273.15)))/(1+10*x[:,2]*x[:,3]*x[:,4]*np.exp(-1000*(1/(x[:,1]+273.15)-1/(40+273.15))))
    return np.array(df)
def data_process_BD(x):
    df = pd.DataFrame(x)
    df[5] = (1.06e20*np.exp(-1.273e4/(x[:,1]+273.15))+1.21e21*np.exp(-1.324e4/(x[:,1]+273.15)))/(2.97e21*np.exp(-1.293e4/(x[:,1]+273.15)) + 1.06e20*np.exp(-1.273e4/(x[:,1]+273.15))+1.21e21*np.exp(-1.324e4/(x[:,1]+273.15)))
    return np.array(df)

# Import Models
import os
os.chdir(os.path.dirname (os.path.realpath (__file__)))
os.chdir(r'model')
def load_model(filename):
    """
    Load a trained model from a file.
    
    Parameters:
    filename (str): The name of the file to load the model from.
    
    Returns:
    model: The loaded model.
    """
    with open(filename, 'rb') as f:
        model = pkl.load(f)
    print(f"Model loaded from {filename}")
    return model

con_model = load_model('conversion.pkl')
scaler_con = load_model('scaler_con.pkl')
BD_model = load_model('BD.pkl')
scaler_BD = load_model('scaler_BD.pkl')

# Optimizer
def objective(x):
    x = x.reshape(1, -1)  
    x = data_process_con(x)
    x = scaler_con.transform(x)
    y = con_model.predict(x)
    if y >= 100:
        y = 100
    elif y <= 0:
        y = 0
    return np.clip(y, -1e4, 1e4)

def restrict(x):
    x = x.reshape(1, -1)  
    x = data_process_BD(x)
    x = scaler_BD.transform(x)
    y = BD_model.predict(x)
    if y <= 0:
        y = 0
    return np.clip(y, -1e4, 1e4)

from scipy.optimize import minimize
def optimization(BD, disp=False, search=False):
    cons = (
        {'type': 'ineq', 'fun': lambda x: restrict(x) - (BD-5)}, # restrict(x) >= BD-5
        {'type': 'ineq', 'fun': lambda x: - restrict(x) + (BD+5)}, # restrict(x) <= BD+5
        {'type': 'ineq', 'fun': lambda x: x[0] - 0.1}, # pressure >= 0.1
        {'type': 'ineq', 'fun': lambda x: - x[0] + 1.3}, # pressure <= 1.3
        {'type': 'ineq', 'fun': lambda x: x[1] + 5}, # temperature >= -5 ℃
        {'type': 'ineq', 'fun': lambda x: - x[1] + 30}, # temperature <= 30 ℃
        {'type': 'ineq', 'fun': lambda x: x[2] - 5}, # time >= 5 min
        {'type': 'ineq', 'fun': lambda x: - x[2] + 17}, # time <= 20 min
        {'type': 'ineq', 'fun': lambda x: x[3] - 20}, # q_Ethylene >= 20
        {'type': 'ineq', 'fun': lambda x: - x[3] + 55}, # q_Ethylene <= 55
        {'type': 'ineq', 'fun': lambda x: x[4] - 0.004}, # c_catalyst >= 0.004
        {'type': 'ineq', 'fun': lambda x: - x[4] + 0.025}, # c_catalyst <= 0.025
        {'type': 'ineq', 'fun': lambda x: objective(x) - 10}, # conversion >= 10%
        {'type': 'ineq', 'fun': lambda x: - objective(x) + 95} # conversion <= 95%
    )

    fun = lambda x: -objective(x)
    temp_con = -1
    best_res = None
    if search:
        print("Searching for the best parameters...")
        for p in [0.1, 0.5, 0.7, 1.2]:
            for T in [-3,5,15,25]:
                for t in [10,15]:
                    for q in [25,35,45]:
                        for c in [0.006, 0.011, 0.016, 0.021]:
                            x0 = np.array([p, T, t, q, c])
                            res = minimize(fun, x0, constraints=cons, method='SLSQP')
                            if res.success and objective(res.x) > temp_con + 0.001:
                                temp_con = objective(res.x)
                                best_res = res
    else:
        print("Optimizing parameters...")
        x0 = np.array([0.8, 10, 15, 45, 0.0122])
        best_res = minimize(fun, x0, constraints=cons, method='SLSQP', options={'maxiter': 1000, 'ftol': 1e-6})
    if disp:
        if best_res is None:
            print("No feasible solution found.")
            return None
        else:
            res = best_res
            print('Success:', res.success)
            print('=' * 30)
            print('Parameters:')
            print('Pressure(MPa):', res.x[0])
            print('Temperature(℃):', res.x[1])
            print('Time(min):', res.x[2])
            print('q_Ethylene(sccm):', res.x[3])
            print('c_catalyst(g/100mL):', res.x[4])
            print('=' * 30)
            print("Results:")
            print('BD:', restrict(res.x))
            print('Conversion(%):', objective(res.x))
            return res

# Design
res = optimization(120, True, False)