import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

systolic_df = pd.read_csv("results/merged_systolic.csv")
systolic_2_node_df = pd.read_csv("results/merged_2_node_systolic.csv")
systolic_n_node_df = pd.read_csv("results/merged_multi_node_systolic.csv")
ref_df = pd.read_csv("results/merged_reference.csv")


# Function to compute the error, the average, the speedup and the efficency
def compute_error(group,full_df):
    total_runs = len(group)                                                                         # Count the number of row in the group
    failed_runs = (group["STATUS"] != 0).sum()                                                      # Count the number of failed runs
    dim, processes = group.name                                                                     # Extract just a value from the group
    avg_time = group["TIME"].mean()                                                                 # Evaluate the avarage of times
    percent_error = (failed_runs / total_runs) * 100                                                # Percentage of error
    ref_avg = group["REF_AVG"].iloc[0]                                                              # Extract just a value from the group
    speedup_ref = ref_avg / avg_time                                                                # Evaluate the speed up compared to the reference
    one_proc_time = full_df[(full_df['DIM'] == dim) & (full_df['PROCS'] == 1)]["TIME"].mean()
    speedup = one_proc_time / avg_time if processes != 1 else 1.0

    return pd.Series({
        "AVG_TIME": avg_time,
        "PERCENT_ERROR": percent_error,
        "TOTAL_RUNS": total_runs,
        "FAILED_RUNS": failed_runs,
        "REF_AVG": ref_avg,
        "SPEEDUP_REF": speedup_ref,
        "SPEEDUP":  speedup,
        "EFFICENCY": speedup/processes * 100
    })


# Merge reference into systolic and summarize
systolic_summary = (
    systolic_df.merge(ref_df[["DIM", "REF_AVG"]], on="DIM")
    .groupby(["DIM", "PROCS"])
    .apply(lambda g: compute_error(g, systolic_df), include_groups=False)
    .reset_index()  # keep DIM and PROCS as columns
)

systolic_2_node_summary = (
    systolic_2_node_df.merge(ref_df[["DIM", "REF_AVG"]], on="DIM")
    .groupby(["DIM", "PROCS"])
    .apply(lambda g: compute_error(g, systolic_df), include_groups=False)
    .reset_index()  # keep DIM and PROCS as columns
)

systolic_n_node_summary = (
    systolic_n_node_df.merge(ref_df[["DIM", "REF_AVG"]], on="DIM")
    .groupby(["DIM", "PROCS"])
    .apply(lambda g: compute_error(g, systolic_df), include_groups=False)
    .reset_index()  # keep DIM and PROCS as columns
)



# Save to CSV
systolic_summary.to_csv("results/final_summary.csv", index=False)
systolic_2_node_summary.to_csv("results/final_2_node_summary.csv", index=False)
systolic_n_node_summary.to_csv("results/final_n_node_summary.csv", index=False)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

print(systolic_summary)
print(systolic_2_node_summary)
print(systolic_n_node_summary)

print("Summary saved.")

# Merge reference into systolic
merged_df = systolic_df.merge(ref_df[["DIM", "REF_AVG"]], on="DIM")
merged_2_node_df = systolic_2_node_df.merge(ref_df[["DIM", "REF_AVG"]], on="DIM")
merged_n_node_df = systolic_n_node_df.merge(ref_df[["DIM", "REF_AVG"]], on="DIM")

# Define models
def linear(x, a, b):
    return a * x + b

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c

def exponential(x, a, b, c):
    # clipping per evitare overflow di exp()
    z = -b * x
    z = np.clip(z, -700, 700)
    return a * np.exp(z) + c

def amdahl(x, a, f, c):
    return a * ((1 - f) + f / x) + c

models = {
    "Linear": (linear, ([-np.inf, -np.inf], [np.inf, np.inf])),
    "Quadratic": (quadratic, ([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])),
    "Exponential": (exponential, ([0, 0, -np.inf], [np.inf, np.inf, np.inf])),  # a ≥ 0, b ≥ 0
    "Amdahl": (amdahl, ([0, 0, -np.inf], [np.inf, 1, np.inf])),  # f ∈ [0,1]
}

param_names = {
    "Linear": ["a", "b"],
    "Quadratic": ["a", "b", "c"],
    "Exponential": ["a", "b", "c"],
    "Amdahl": ["a", "f", "c"],
}


dims = merged_df["DIM"].unique()

for dim in dims:
    subset = merged_df[merged_df["DIM"] == dim]

    process_counts = sorted(subset["PROCS"].unique())
    data = [subset[subset["PROCS"] == p]["TIME"].values for p in process_counts]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, positions=process_counts)
    plt.axhline(subset["REF_AVG"].iloc[0], color="red", linestyle="--", label="Reference AVG")

    x = np.array(process_counts)
    y = np.array([np.mean(d) for d in data])

    best_model = None
    best_r2 = -np.inf
    y_fit_best = None
    best_param = None

    fits = {}

    for name, (model, bounds) in models.items():
        try:
            popt, _ = curve_fit(model, x, y, bounds=bounds, maxfev=20000)
            y_fit = model(x, *popt)
            r2 = r2_score(y, y_fit)
            fits[name] = (popt, y_fit, r2)

            if r2 > best_r2:
                best_r2 = r2
                best_model = name
                best_param = popt
                y_fit_best = y_fit
        except RuntimeError:
            print(f"Fit did not converge for {name}")

    if y_fit_best is not None:
        names = param_names.get(best_model, [f"p{i}" for i in range(len(best_param))])
        param_str = ", ".join([f"{name}={p:.2f}" for name, p in zip(names, best_param)])
        plt.plot(
            x, y_fit_best, "-",
            label=f'Best: {best_model} ({param_str}, R²={best_r2:.3f})'
        )

    if "Amdahl" in fits and best_model != "Amdahl":
        popt_a, y_fit_a, r2_a = fits["Amdahl"]
        a_val, f_val, c_val = popt_a
        plt.plot(
            x, y_fit_a, "--",
            label=f'Amdahl (a={a_val:.2f}, f={f_val:.2f}, c={c_val:.2f}, R²={r2_a:.3f})'
        )

    plt.title(f"Execution Time Distribution per process (DIM={dim})")
    plt.xlabel("Number of Processes")
    plt.ylabel("Execution Time (s)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(process_counts)
    plt.tight_layout()
    plt.show()

processes_to_plot = [1,8,16,24, 32, 40, 48,56,64,72]  # example process counts; adjust as needed

plt.figure(figsize=(8, 5))

for p in processes_to_plot:
    subset = systolic_summary[systolic_summary["PROCS"] == p].sort_values("DIM")
    plt.plot(subset["DIM"], subset["AVG_TIME"], marker="o", label=f"{p} processes")

plt.title("Execution Time vs Matrix Order for Different processes")
plt.xlabel("DIM")
plt.ylabel("Execution Time (s)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Filter for DIM=500
dim_value = 500
single_node = systolic_summary[systolic_summary["DIM"] == dim_value]
two_node = systolic_2_node_summary[systolic_2_node_summary["DIM"] == dim_value]
multi_node = systolic_n_node_summary[systolic_n_node_summary["DIM"] == dim_value]

plt.figure(figsize=(10,6))

# Plot single-node
plt.plot(single_node["PROCS"], single_node["AVG_TIME"], marker='o', label="Single Node")

# Plot 2-node
plt.plot(two_node["PROCS"], two_node["AVG_TIME"], marker='s', label="2 Nodes")

# Plot n-node
plt.plot(multi_node["PROCS"], multi_node["AVG_TIME"], marker='^', label="Multi Node")

plt.title(f"Execution Time Comparison (Matrix order={dim_value})")
plt.xlabel("Number of Processes")
plt.ylabel("Average Execution Time (s)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
