import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 1. Load Excel Dataset
# ----------------------------
file_path = r"E:\AI Projects\My Project\Files\aggr.xlsx"     # <- change this
sheet_name = "Golpakhsh"
df = pd.read_excel(file_path, sheet_name=sheet_name)

print("=== FIRST 5 ROWS ===")
print(df.head())

print("\n=== DATA SUMMARY ===")
print(df.info())


# ----------------------------
# 2. Descriptive Statistics
# ----------------------------
print("\n=== DESCRIPTIVE STATISTICS ===")
print(df.describe(include='all'))


# ----------------------------
# 3. One-Way ANOVA
# ----------------------------
group_column = "CompanyName"     # <- change this
value_column = "NetAmount"     # <- change this

# Split data into groups
groups = [group[value_column].dropna().values 
          for name, group in df.groupby(group_column)]

if len(groups) >= 2:
    f_stat, p_value = stats.f_oneway(*groups)
    print("\n=== ONE-WAY ANOVA ===")
    print(f"F-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.6f}")
else:
    print("\nNot enough groups for ANOVA.")


# ----------------------------
# 4. Statistical Distribution
# ----------------------------
numeric_cols = df.select_dtypes(include='number').columns

print("\n=== DISTRIBUTION ANALYSIS ===")

dist_summary = {}

for col in numeric_cols:
    data = df[col].dropna()

    # Skewness and kurtosis
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)

    # Normality test
    shapiro_stat, shapiro_p = stats.shapiro(data)

    dist_summary[col] = {
        "skewness": skewness,
        "kurtosis": kurtosis,
        "shapiro_statistic": shapiro_stat,
        "shapiro_p_value": shapiro_p
    }

    print(f"\n--- {col} ---")
    print(f"Skewness: {skewness:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")
    print(f"Shapiro-Wilk p-value: {shapiro_p:.6f}")
    
    # Plot Histogram + KDE
    plt.figure(figsize=(7,4))
    sns.histplot(data, kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.show()

    # QQ Plot (optional)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f"QQ Plot for {col}")
    plt.show()


# ----------------------------
# 5. Summary for LLM
# ----------------------------
summary = {
    "columns": list(df.columns),
    "num_rows": len(df),
    "descriptive_stats": df.describe(include='all').to_dict(),
    "distribution": dist_summary
}

print("\n=== SUMMARY FOR LLM (JSON-LIKE) ===")
print(summary)
