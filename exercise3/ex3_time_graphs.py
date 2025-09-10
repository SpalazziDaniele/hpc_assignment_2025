
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


x = []
y = []

with open("/Users/enricobozzetto/Desktop/PoliTO/1year/HPC/assignment/graphs/50/execution_times_A_50.csv", 'r', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for idx, row in enumerate(reader):
        if not row:
            continue
        try:
            valore = float(row[0])
        except ValueError:
            continue
        x.append(idx+1)
        y.append(valore)
x.pop(-1)
y.pop(-1)

x = np.array(x)
y = np.array(y)


def linear(x, a, b):
    return a * x + b

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def exponential(x, a, b, c):
    return a * np.exp(-b * x) + c

def amdahl(x, a, f, c):
    return a * ((1 - f) + f / x) + c


models = {
    'Linear': linear,
    'Quadratic': quadratic,
    'Exponential': exponential,
    'Amdahl': amdahl,
}

best_model = None
best_r2 = -np.inf
y_fit_best = None
best_param=None


for name, model in models.items():
    try:
        popt, _ = curve_fit(model, x, y, maxfev=10000)
        y_fit = model(x, *popt)
        r2 = r2_score(y, y_fit)
        if r2 > best_r2:
            best_r2 = r2
            best_model = name
            best_param = popt
            y_fit_best = y_fit
    except RuntimeError:
        print(f"Fit did not converge for {name}")


plt.figure(figsize=(10, 5))
plt.plot(x, y, 'o', label='Data')

if y_fit_best is not None:
    param_str = ", ".join([f"{p:.2f}" for p in best_param])
    plt.plot(x, y_fit_best, '-', 
             label=f'Best model: {best_model} (params={param_str}, R²={best_r2:.3f})')

plt.xticks(x)
plt.xlabel('Threads')
plt.ylabel('Time [s]')
plt.title('Execution times - Model fitting')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
