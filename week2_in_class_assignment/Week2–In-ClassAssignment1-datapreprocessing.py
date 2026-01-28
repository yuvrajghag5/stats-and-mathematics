import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore



# Step 1

df = pd.read_csv(r"C:\Users\yuvra\OneDrive\Desktop\Codex\stats_and_machine_learning\code\Titanic-Dataset.csv")

#replacing the missing values
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

#dropping Cabin column because it containes heavily missing data.

df.drop(columns =['Cabin'], inplace = True )

print("\n after replacing \n")
print(df.isnull().sum())
print(f"\n {df['Embarked'].head(10)}")

# Explanation
# Missing values were identified in the Age, Embarked, and Cabin columns. Age (Numerical feature) missing values were replaced using the median.
# Embarked(categorial features)missing values were replaced using the mode. Cabin column was dropped because it contains a 
# very high percentage of missing values




# Step 2

# Adding artificial Noise
np.random.seed(42)
df['noisy_age'] = df['Age'] + np.random.normal(0, 2, size=len(df))

# Handling noisy data 
df['smooth_age'] = df['noisy_age'].rolling(window = 10, min_periods = 1).mean()
print(df[['Age','noisy_age', 'smooth_age']].head(10))


# Explanation
# The Age column was selected as the numerical feature. Artificial noise was added using small random values from a normal distribution.
# This shows that the real world data is imperfect and measurement errors. A moving average smoothing technique was applied using a rolling window.
# Comparison showed that smoothed values were more stable than noisy values.



# Step 3
age_z = zscore(df['Age']) 
outlier = df[abs(age_z) > 3]
print(outlier[['Age']].head())

df = df[abs(age_z) <= 3]
print(f"\n {df['Age'].head(100)}")


# Explanation
# Data points with absolute Z-score greater than 3 were treated as outliers. Identified outliers were removed from the dataset.
# Removal was chosen to prevent extreme values from affecting analysis and model performance.




# Step 4
print("\n one hot encoding \n")
df_encoded = pd.get_dummies(df, columns = ['Embarked'], drop_first = False)
print(df_encoded.head())


# Explanation
# One-hot encoding was applied to the Embarked categorical feature.Machine learning models require numerical input.
# One-hot encoding converts categories into binary columns.This prevents incorrect ordinal relationships between categories.
# The transformation enables categorical data to be used effectively in models.


# Step 5

features = df[['Age', 'Fare']]
std = StandardScaler()
min_max = MinMaxScaler()

df_std = std.fit_transform(features)
df_mm = min_max.fit_transform(features)


df_std = pd.DataFrame(df_std, columns = ['age_std', 'fare_std'])
df_mm = pd.DataFrame(df_mm, columns = ['age_mm', 'fare_mm'])

print(f"\n z standardization: \n{df_std.head(10)}")
print(f"\n min max standardization: \n{df_mm.head(10)}")


# # Explanation 
# Z-score Standardization: Preferred for algorithms sensitive to variance (SVM, KNN, Logistic Regression).Suitable when data follows a normal distribution.
# Min–Max Normalization: Preferred for neural networks and bounded input requirements. Preserves relative distances between values.