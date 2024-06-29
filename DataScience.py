import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from IPython.display import display

childMortality = pd.read_csv(r"C:\Users\terzh\Downloads\1211103705_DATA SCIENCE FUNDAMENTALS\DataSci Assignment\datasets\child-mortality-rate-vs-mean-male-height-cm.csv")
HDI = pd.read_csv(r"C:\Users\terzh\Downloads\1211103705_DATA SCIENCE FUNDAMENTALS\DataSci Assignment\datasets\human-development-index-vs-mean-male-height.csv")
cropYields = pd.read_csv(r"C:\Users\terzh\Downloads\1211103705_DATA SCIENCE FUNDAMENTALS\DataSci Assignment\datasets\key-crop-yields.csv")
caloricSupply = pd.read_csv(r"C:\Users\terzh\Downloads\1211103705_DATA SCIENCE FUNDAMENTALS\DataSci Assignment\datasets\daily-per-capita-caloric-supply.csv")
proteinSupply = pd.read_csv(r"C:\Users\terzh\Downloads\1211103705_DATA SCIENCE FUNDAMENTALS\DataSci Assignment\datasets\daily-per-capita-protein-supply.csv")
fatSupply = pd.read_csv(r"C:\Users\terzh\Downloads\1211103705_DATA SCIENCE FUNDAMENTALS\DataSci Assignment\datasets\daily-per-capita-fat-supply.csv")
eggConsumption = pd.read_csv(r"C:\Users\terzh\Downloads\1211103705_DATA SCIENCE FUNDAMENTALS\DataSci Assignment\datasets\per-capita-egg-consumption-kilograms-per-year.csv")
meatConsumption = pd.read_csv(r"C:\Users\terzh\Downloads\1211103705_DATA SCIENCE FUNDAMENTALS\DataSci Assignment\datasets\per-capita-meat-consumption-by-type-kilograms-per-year.csv")
milkConsumption = pd.read_csv(r"C:\Users\terzh\Downloads\1211103705_DATA SCIENCE FUNDAMENTALS\DataSci Assignment\datasets\per-capita-milk-consumption.csv")
animalProtein = pd.read_csv(r"C:\Users\terzh\Downloads\1211103705_DATA SCIENCE FUNDAMENTALS\DataSci Assignment\datasets\share-of-calories-from-animal-protein-vs-mean-male-height.csv")

childMortalityCor = childMortality.drop(["Code", "Continent"], axis=1)
HDICor = HDI.drop(["Code", "Continent", "Mean male height (cm)"], axis=1)
cropYieldsCor = cropYields.drop(["Code"], axis=1)
caloricSupplyCor = caloricSupply.drop(["Code"], axis=1)
proteinSupplyCor = proteinSupply.drop(["Code"], axis=1)
fatSupplyCor = fatSupply.drop(["Code"], axis=1)
eggConsumptionCor = eggConsumption.drop(["Code"], axis=1)
meatConsumptionCor = meatConsumption.drop(["Code"], axis=1)
milkConsumptionCor = milkConsumption.drop(["Code"], axis=1)
animalProteinCor = animalProtein.drop(["Code", "Continent"], axis=1)

HDICor

dfs = [childMortalityCor, HDICor, cropYieldsCor, caloricSupplyCor, proteinSupplyCor, fatSupplyCor, 
       eggConsumptionCor, meatConsumptionCor, milkConsumptionCor, animalProteinCor]
mergeResult = reduce(lambda  left,right: pd.merge(left,right,on = ["Entity", "Year"]), dfs)
mergeResult = mergeResult.drop(["Mean male height (cm)_y", "Population (historical estimates)"], axis=1)

mergeResult = mergeResult.rename(columns={'Observation value - Unit of measure: Deaths per 100 live births - Indicator: Under-five mortality rate - Sex: Both sexes - Wealth quintile: All wealth quintiles': 'Deaths per 100 births'})
mergeResult = mergeResult.rename(columns={'Daily calorie supply per person that comes from animal protein': 'Daily calorie supply from animal protein'})
mergeResult = mergeResult.rename(columns={'Mean male height (cm)_x': 'Mean male height (cm)'})
mergeResult = mergeResult.rename(columns={'Wheat | 00000015 || Yield | 005419 || tonnes per hectare': 'Wheat Yields (tonnes per hectare)'})
mergeResult = mergeResult.rename(columns={'Rice | 00000027 || Yield | 005419 || tonnes per hectare': 'Rice Yields (tonnes per hectare)'})
mergeResult = mergeResult.rename(columns={'Bananas | 00000486 || Yield | 005419 || tonnes per hectare': 'Banana Yields (tonnes per hectare)'})
mergeResult = mergeResult.rename(columns={'Maize | 00000056 || Yield | 005419 || tonnes per hectare': 'Maize Yields (tonnes per hectare)'})
mergeResult = mergeResult.rename(columns={'Soybeans | 00000236 || Yield | 005419 || tonnes per hectare': 'Soybean Yields (tonnes per hectare)'})
mergeResult = mergeResult.rename(columns={'Potatoes | 00000116 || Yield | 005419 || tonnes per hectare': 'Potato Yields (tonnes per hectare)'})
mergeResult = mergeResult.rename(columns={'Beans, dry | 00000176 || Yield | 005419 || tonnes per hectare': 'Bean Yields (tonnes per hectare)'})
mergeResult = mergeResult.rename(columns={'Peas, dry | 00000187 || Yield | 005419 || tonnes per hectare': 'Pea Yields (tonnes per hectare)'})
mergeResult = mergeResult.rename(columns={'Cassava | 00000125 || Yield | 005419 || tonnes per hectare': 'Cassava Yields (tonnes per hectare)'})
mergeResult = mergeResult.rename(columns={'Cocoa beans | 00000661 || Yield | 005419 || tonnes per hectare': 'Cocoa Yields (tonnes per hectare)'})
mergeResult = mergeResult.rename(columns={'Barley | 00000044 || Yield | 005419 || tonnes per hectare': 'Barley Yields (tonnes per hectare)'})
mergeResult = mergeResult.rename(columns={'Daily calorie supply per person': 'Calorie supply (per day per capita)'})
mergeResult = mergeResult.rename(columns={'Total | 00002901 || Food available for consumption | 0674pc || grams of protein per day per capita': 'Protein supply (grams per day per capita)'})
mergeResult = mergeResult.rename(columns={'Total | 00002901 || Food available for consumption | 0684pc || grams of fat per day per capita': 'Fat supply (grams per day per capita)'})
mergeResult = mergeResult.rename(columns={'Eggs | 00002949 || Food available for consumption | 0645pc || kilograms per year per capita': 'Egg consumption (kilograms per year per capita)'})
mergeResult = mergeResult.rename(columns={'Meat, Other | 00002735 || Food available for consumption | 0645pc || kilograms per year per capita': 'Other meats consumption (kilograms per year per capita)'})
mergeResult = mergeResult.rename(columns={'Meat, sheep and goat | 00002732 || Food available for consumption | 0645pc || kilograms per year per capita': 'Sheep and goat consumption (kilograms per year per capita)'})
mergeResult = mergeResult.rename(columns={'Meat, beef | 00002731 || Food available for consumption | 0645pc || kilograms per year per capita': 'Beef consumption (kilograms per year per capita)'})
mergeResult = mergeResult.rename(columns={'Meat, pig | 00002733 || Food available for consumption | 0645pc || kilograms per year per capita': 'Pig consumption (kilograms per year per capita)'})
mergeResult = mergeResult.rename(columns={'Meat, poultry | 00002734 || Food available for consumption | 0645pc || kilograms per year per capita': 'Poultry consumption (kilograms per year per capita)'})
mergeResult = mergeResult.rename(columns={'Milk - Excluding Butter | 00002848 || Food available for consumption | 0645pc || kilograms per year per capita': 'Milk consumption (kilograms per year per capita)'})
mergeResult

result1 = mergeResult.drop(["Human Development Index"], axis=1)
result1 = result1[result1["Mean male height (cm)"].notna()]
result1

#Can skip this part if you save the data
result1.to_csv('Height(CleanData).csv', index=False)

result2 = mergeResult.drop(["Mean male height (cm)"], axis=1)
result2 = result2[result2["Human Development Index"].notna()]
result2

result2.to_csv('HDI(CleanData).csv', index=False)

#Rerun the clean datas
height_data = pd.read_csv(r"C:\Users\terzh\Downloads\1211103705_DATA SCIENCE FUNDAMENTALS\DataSci Assignment\Clean_dataset\Height(CleanData).csv")
hdi_data  = pd.read_csv(r"C:\Users\terzh\Downloads\1211103705_DATA SCIENCE FUNDAMENTALS\DataSci Assignment\Clean_dataset\HDI(CleanData).csv")

df_merged = pd.merge(height_data, hdi_data, on=['Entity', 'Year'], how='inner')

# Inspect column names
print(df_merged.columns)

# Correct the column names based on inspection
# Assuming 'Deaths per 100 births' is actually 'Deaths per 100 births_x'
mean_deaths_per_100_births = df_merged['Deaths per 100 births_x'].mean()
df_merged['Deaths per 100 births_x'] = df_merged['Deaths per 100 births_x'].fillna(mean_deaths_per_100_births)

# Impute other relevant columns with their mean values if necessary
macronutrient_columns = ['Calorie supply (per day per capita)_x', 'Protein supply (grams per day per capita)_x', 'Fat supply (grams per day per capita)_x']
for col in macronutrient_columns:
    df_merged[col] = df_merged[col].fillna(df_merged[col].mean())

# Step 4: Correlation analysis
correlation_matrix = df_merged[['Mean male height (cm)', 'Deaths per 100 births_x', 'Human Development Index'] + macronutrient_columns].corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Impute missing values with mean
mean_deaths_per_100_births = df_merged['Deaths per 100 births_x'].mean()
df_merged['Deaths per 100 births_x'].fillna(mean_deaths_per_100_births, inplace=True)

# Recalculate missing values after imputation
missing_values = df_merged.isnull().sum()
print("Missing Values after mean imputation:")
print(missing_values)


display(df_merged.head(), missing_values)
# Analyze correlations again
correlation_matrix = df_merged[['Mean male height (cm)', 'Deaths per 100 births_x', 'Human Development Index']].corr()
print("\nCorrelation Matrix after mean imputation:")
print(correlation_matrix)


# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

# Step 5: Scatter plots with regression lines
plt.figure(figsize=(10, 6))
sns.regplot(x='Protein supply (grams per day per capita)_x', y='Mean male height (cm)', data=df_merged, scatter_kws={'alpha':0.5})
plt.title('Protein Supply vs Mean Male Height')
plt.xlabel('Protein supply (grams per day per capita)')
plt.ylabel('Mean male height (cm)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(x='Fat supply (grams per day per capita)_x', y='Deaths per 100 births_x', data=df_merged, scatter_kws={'alpha':0.5})
plt.title('Fat Supply vs Infant Mortality')
plt.xlabel('Fat supply (grams per day per capita)')
plt.ylabel('Deaths per 100 births_x')
plt.grid(True)
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Assuming df_merged is your merged DataFrame with relevant columns
# Ensure df_merged contains columns related to features (X) and target variable (y)

# Define features (X) and target variable (y) for Physical Growth Patterns (e.g., Mean male height)
X_growth = df_merged[['Protein supply (grams per day per capita)_x']]
y_growth = df_merged['Mean male height (cm)']

# Split the data into training and testing sets (80% training, 20% testing)
X_growth_train, X_growth_test, y_growth_train, y_growth_test = train_test_split(X_growth, y_growth, test_size=0.2, random_state=42)

# Initialize and train the linear regression model for Physical Growth Patterns
model_growth = LinearRegression()
model_growth.fit(X_growth_train, y_growth_train)

# Make predictions on the test set
y_growth_pred = model_growth.predict(X_growth_test)

# Evaluate the model for Physical Growth Patterns
mse_growth = mean_squared_error(y_growth_test, y_growth_pred)
r2_growth = r2_score(y_growth_test, y_growth_pred)

# Display the model coefficients, mean squared error, and R^2 score for Physical Growth Patterns
print("Physical Growth Patterns:")
print(f"Coefficient: {model_growth.coef_[0]}")
print(f"Mean Squared Error: {mse_growth:.2f}")
print(f"R^2 Score: {r2_growth:.2f}")
print()

# Define features (X) and target variable (y) for Infant Mortality (e.g., Deaths per 100 births)
X_mortality = df_merged[['Fat supply (grams per day per capita)_x']]
y_mortality = df_merged['Deaths per 100 births_x']

# Split the data into training and testing sets (80% training, 20% testing)
X_mortality_train, X_mortality_test, y_mortality_train, y_mortality_test = train_test_split(X_mortality, y_mortality, test_size=0.2, random_state=42)

# Initialize and train the linear regression model for Infant Mortality
model_mortality = LinearRegression()
model_mortality.fit(X_mortality_train, y_mortality_train)

# Make predictions on the test set
y_mortality_pred = model_mortality.predict(X_mortality_test)

# Evaluate the model for Infant Mortality
mse_mortality = mean_squared_error(y_mortality_test, y_mortality_pred)
r2_mortality = r2_score(y_mortality_test, y_mortality_pred)

# Display the model coefficients, mean squared error, and R^2 score for Infant Mortality
print("Infant Mortality:")
print(f"Coefficient: {model_mortality.coef_[0]}")
print(f"Mean Squared Error: {mse_mortality:.2f}")
print(f"R^2 Score: {r2_mortality:.2f}")

#Sub Question 2

# Select relevant columns for the analysis
height_data_selected = height_data[['Entity', 'Year', 'Protein supply (grams per day per capita)', 'Fat supply (grams per day per capita)', 'Calorie supply (per day per capita)']]
hdi_data_selected = hdi_data[['Entity', 'Year', 'Human Development Index']]

# Merge the datasets on 'Entity' and 'Year'
combined_data = pd.merge(height_data_selected, hdi_data_selected, on=['Entity', 'Year'])

# Check for missing values
missing_values = combined_data.isnull().sum()
print(missing_values)

display(combined_data.head(), missing_values)

# Select only numeric columns for the correlation matrix
numeric_columns = combined_data.select_dtypes(include=['number'])

# Correlation matrix
correlation_matrix = numeric_columns.corr()
display("Correlation Matrix:", correlation_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix between Macronutrient Intake and HDI')
plt.show()

# Scatter plot with regression line: Protein supply vs. HDI
plt.figure(figsize=(8, 6))
sns.regplot(x='Protein supply (grams per day per capita)', y='Human Development Index', data=combined_data, scatter_kws={'alpha':0.5})
plt.title('Protein Supply vs. Human Development Index')
plt.xlabel('Protein Supply (grams per day per capita)')
plt.ylabel('Human Development Index')
plt.show()

# Scatter plot with regression line: Fat supply vs. HDI
plt.figure(figsize=(8, 6))
sns.regplot(x='Fat supply (grams per day per capita)', y='Human Development Index', data=combined_data, scatter_kws={'alpha':0.5})
plt.title('Fat Supply vs. Human Development Index')
plt.xlabel('Fat Supply (grams per day per capita)')
plt.ylabel('Human Development Index')
plt.show()

# Scatter plot with regression line: Calorie supply vs. HDI
plt.figure(figsize=(8, 6))
sns.regplot(x='Calorie supply (per day per capita)', y='Human Development Index', data=combined_data, scatter_kws={'alpha':0.5})
plt.title('Calorie Supply vs. Human Development Index')
plt.xlabel('Calorie Supply (per day per capita)')
plt.ylabel('Human Development Index')
plt.show()

# Define the features and the target variable
X = combined_data[['Protein supply (grams per day per capita)', 'Fat supply (grams per day per capita)', 'Calorie supply (per day per capita)']]
y = combined_data['Human Development Index']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the model coefficients, mean squared error, and R^2 score
model_coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("Model Coefficients:\n")
display(model_coefficients)

model_performance = pd.DataFrame({'Mean Squared Error': [mse], 'R^2 Score': [r2]})
print("Model Coefficients:\n")
display(model_performance)

# Scatter plot of actual vs predicted HDI values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Actual HDI')
plt.ylabel('Predicted HDI')
plt.title('Actual vs Predicted HDI')
plt.show()

#Sub Question 3

display(height_data.head())
display(hdi_data.head())

height_data_selected = height_data[['Entity', 'Year', 'Mean male height (cm)', 'Wheat Yields (tonnes per hectare)', 
                                    'Rice Yields (tonnes per hectare)', 'Banana Yields (tonnes per hectare)', 
                                    'Maize Yields (tonnes per hectare)', 'Soybean Yields (tonnes per hectare)', 
                                    'Potato Yields (tonnes per hectare)', 'Deaths per 100 births']]

hdi_data_selected = hdi_data[['Entity', 'Year', 'Human Development Index']]

combined_data = pd.merge(height_data_selected, hdi_data_selected, on=['Entity', 'Year'])

# Rename columns for easier referencing
combined_data.rename(columns={
    'Mean male height (cm)': 'height',
    'Wheat Yields (tonnes per hectare)': 'wheat_yield',
    'Rice Yields (tonnes per hectare)': 'rice_yield',
    'Banana Yields (tonnes per hectare)': 'banana_yield',
    'Maize Yields (tonnes per hectare)': 'maize_yield',
    'Soybean Yields (tonnes per hectare)': 'soybean_yield',
    'Potato Yields (tonnes per hectare)': 'potato_yield',
    'Deaths per 100 births': 'infant_mortality'
}, inplace=True)

combined_data.fillna(combined_data.select_dtypes(include=[float, int]).mean(), inplace=True)

# Display the first few rows to verify
display(combined_data.head())

display(combined_data.describe())

plt.figure(figsize=(14, 8))
sns.histplot(combined_data['infant_mortality'], kde=True)
plt.title('Distribution of Infant Mortality')
plt.xlabel('Infant Mortality (Deaths per 100 births)')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(14, 8))
sns.histplot(combined_data['height'], kde=True)
plt.title('Distribution of Mean Male Height')
plt.xlabel('Height (cm)')
plt.ylabel('Frequency')
plt.show()

# Correlation matrix
numeric_columns = combined_data.select_dtypes(include=['number'])
correlation_matrix = numeric_columns.corr()
print("Correlation Matrix:")
display(correlation_matrix)

# Heatmap of the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix between Crop Yields and Health Indicators')
plt.show()

# Define the independent variables (crop yields) and dependent variables (height, infant mortality)
X = combined_data[['wheat_yield', 'rice_yield', 'banana_yield', 'maize_yield', 'soybean_yield', 'potato_yield']]
X = sm.add_constant(X)  # Add a constant term for the intercept

# Dependent variable: height
y_height = combined_data['height']
model_height = sm.OLS(y_height, X).fit()
display(model_height.summary())

# Dependent variable: infant_mortality
y_infant_mortality = combined_data['infant_mortality']
model_infant_mortality = sm.OLS(y_infant_mortality, X).fit()
display(model_infant_mortality.summary())

#Sub Question 4

# Select relevant columns for the analysis
height_data_selected = height_data[['Entity', 'Year', 'Wheat Yields (tonnes per hectare)', 'Rice Yields (tonnes per hectare)', 
                                    'Banana Yields (tonnes per hectare)', 'Maize Yields (tonnes per hectare)', 
                                    'Soybean Yields (tonnes per hectare)', 'Potato Yields (tonnes per hectare)']]
hdi_data_selected = hdi_data[['Entity', 'Year', 'Human Development Index']]

# Merge the datasets on 'Entity' and 'Year'
combined_data = pd.merge(height_data_selected, hdi_data_selected, on=['Entity', 'Year'])

# Impute missing values with the mean for numerical columns
combined_data_imputed = combined_data.copy()
numeric_columns = combined_data_imputed.select_dtypes(include=['number']).columns
combined_data_imputed[numeric_columns] = combined_data_imputed[numeric_columns].fillna(combined_data_imputed[numeric_columns].mean())

combined_data_imputed.head()

# Select only numeric columns for the correlation matrix
numeric_columns = combined_data_imputed.select_dtypes(include=['number'])

# Correlation matrix
correlation_matrix = numeric_columns.corr()
display(correlation_matrix)

# Visualize the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix between Crop Yields and HDI')
plt.show()

# Scatter plot with regression line: Wheat Yields vs. HDI
plt.figure(figsize=(8, 6))
sns.regplot(x='Wheat Yields (tonnes per hectare)', y='Human Development Index', data=combined_data_imputed, scatter_kws={'alpha':0.5})
plt.title('Wheat Yields vs. Human Development Index')
plt.xlabel('Wheat Yields (tonnes per hectare)')
plt.ylabel('Human Development Index')
plt.show()

# Scatter plot with regression line: Rice Yields vs. HDI
plt.figure(figsize=(8, 6))
sns.regplot(x='Rice Yields (tonnes per hectare)', y='Human Development Index', data=combined_data_imputed, scatter_kws={'alpha':0.5})
plt.title('Rice Yields vs. Human Development Index')
plt.xlabel('Rice Yields (tonnes per hectare)')
plt.ylabel('Human Development Index')
plt.show()

# Scatter plot with regression line: Banana Yields vs. HDI
plt.figure(figsize=(8, 6))
sns.regplot(x='Banana Yields (tonnes per hectare)', y='Human Development Index', data=combined_data_imputed, scatter_kws={'alpha':0.5})
plt.title('Banana Yields vs. Human Development Index')
plt.xlabel('Banana Yields (tonnes per hectare)')
plt.ylabel('Human Development Index')
plt.show()

# Scatter plot with regression line: Maize Yields vs. HDI
plt.figure(figsize=(8, 6))
sns.regplot(x='Maize Yields (tonnes per hectare)', y='Human Development Index', data=combined_data_imputed, scatter_kws={'alpha':0.5})
plt.title('Maize Yields vs. Human Development Index')
plt.xlabel('Maize Yields (tonnes per hectare)')
plt.ylabel('Human Development Index')
plt.show()

# Scatter plot with regression line: Soybean Yields vs. HDI
plt.figure(figsize=(8, 6))
sns.regplot(x='Soybean Yields (tonnes per hectare)', y='Human Development Index', data=combined_data_imputed, scatter_kws={'alpha':0.5})
plt.title('Soybean Yields vs. Human Development Index')
plt.xlabel('Soybean Yields (tonnes per hectare)')
plt.ylabel('Human Development Index')
plt.show()

# Scatter plot with regression line: Potato Yields vs. HDI
plt.figure(figsize=(8, 6))
sns.regplot(x='Potato Yields (tonnes per hectare)', y='Human Development Index', data=combined_data_imputed, scatter_kws={'alpha':0.5})
plt.title('Potato Yields vs. Human Development Index')
plt.xlabel('Potato Yields (tonnes per hectare)')
plt.ylabel('Human Development Index')
plt.show()

# Define the features and the target variable
X = combined_data_imputed[['Wheat Yields (tonnes per hectare)', 'Rice Yields (tonnes per hectare)', 
                           'Banana Yields (tonnes per hectare)', 'Maize Yields (tonnes per hectare)', 
                           'Soybean Yields (tonnes per hectare)', 'Potato Yields (tonnes per hectare)']]
y = combined_data_imputed['Human Development Index']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the model coefficients, mean squared error, and R^2 score
model_coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("Model Coefficients:")
display(model_coefficients)

model_performance = pd.DataFrame({'Mean Squared Error': [mse], 'R^2 Score': [r2]})
print("Model Performance:")
display(model_performance)
