import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load your dataset (update the path/filename as needed)
df = pd.read_csv(r"C:\Users\Meghana D Hegde\Downloads\dataprocessing\Titanic-Dataset.csv")  # Updated to actual data file

# 1. Summary statistics
desc = df.describe(include='all')
print('Summary statistics:')
print(desc)
print('\nMedian values:')
print(df.median(numeric_only=True))

# 2. Histograms and boxplots for numeric features
num_cols = df.select_dtypes(include='number').columns
for col in num_cols:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col}')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

# 3. Pairplot and correlation matrix
sns.pairplot(df[num_cols])
plt.suptitle('Pairplot of Numeric Features', y=1.02)
plt.show()

corr = df[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# 4. Interactive visualization (optional, with Plotly)
# px.scatter_matrix(df[num_cols], title='Scatter Matrix (Plotly)').show()

# 5. Inference comments:
# - Review the summary statistics and plots above to identify outliers, skewness, and relationships.
# - Look for strong correlations, unusual distributions, or anomalies in the data.
# - Use these insights to guide further feature engineering or data cleaning steps.