# Analyzing Data with Pandas and Visualizing Results with Matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# ------------------------------
# Task 1: Load and Explore Dataset
# ------------------------------

# Option 1: Load from CSV (with error handling)
file_path = "iris.csv"  # replace with your file name if using CSV

try:
    df = pd.read_csv(file_path)
    print(f"✅ Successfully loaded dataset from {file_path}")
except FileNotFoundError:
    print(f"⚠ File not found: {file_path}. Loading Iris dataset from sklearn instead...")
    iris = load_iris(as_frame=True)
    df = iris.frame
    df['species'] = df['target'].map(dict(enumerate(iris.target_names)))

# Display first few rows
print("\nFirst 5 rows of the dataset:")
display(df.head())

# Check dataset info
print("\nDataset Info:")
print(df.info())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Clean dataset (fill missing values if any)
df = df.fillna(df.mean(numeric_only=True))

# ------------------------------
# Task 2: Basic Data Analysis
# ------------------------------
print("\nDescriptive statistics:")
display(df.describe())

# Grouping by species if available
if 'species' in df.columns:
    group_means = df.groupby('species').mean()
    print("\nMean values grouped by species:")
    display(group_means)

# ------------------------------
# Task 3: Data Visualization
# ------------------------------

# 1. Line Chart
plt.figure(figsize=(8, 5))
plt.plot(df.index, df.iloc[:, 0], label=df.columns[0])
plt.title("Line Chart - First Numerical Feature Across Samples")
plt.xlabel("Sample Index")
plt.ylabel(df.columns[0])
plt.legend()
plt.show()

# 2. Bar Chart
if 'species' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.barplot(x='species', y=df.columns[1], data=df, estimator='mean')
    plt.title(f"Bar Chart - Average {df.columns[1]} per Species")
    plt.xlabel("Species")
    plt.ylabel(f"Avg {df.columns[1]}")
    plt.show()

# 3. Histogram
plt.figure(figsize=(8, 5))
plt.hist(df.iloc[:, 2], bins=15, color='skyblue', edgecolor='black')
plt.title(f"Histogram - {df.columns[2]} Distribution")
plt.xlabel(df.columns[2])
plt.ylabel("Frequency")
plt.show()

# 4. Scatter Plot
if 'species' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df.columns[0], y=df.columns[2], hue='species', data=df, palette='Set1')
    plt.title(f"Scatter Plot - {df.columns[0]} vs {df.columns[2]}")
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[2])
    plt.legend(title="Species")
    plt.show()

# Halleluyahhh! We have successfully completed the tasks of loading, analyzing, and visualizing the dataset using Pandas and Matplotlib.