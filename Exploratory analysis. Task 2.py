import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = {
    'employee_id': range(1, 16),

    'age': [25, 30, 28, 35, 40, 26, 32, 29, 45, 38, 27, 31, 36, 42, 33],

    'department': [
        'IT', 'HR', 'Finance', 'IT', 'Sales',
        'HR', 'Finance', 'IT', 'Sales', 'Finance',
        'HR', 'Sales', 'IT', 'Finance', 'Sales'
    ],

    'city': [
        'Chennai', 'Bangalore', 'Chennai', 'Mumbai', 'Bangalore',
        'Chennai', 'Mumbai', 'Bangalore', 'Chennai', 'Mumbai',
        'Bangalore', 'Chennai', 'Mumbai', 'Bangalore', 'Chennai'
    ],

    'gender': [
        'Male', 'Female', 'Male', 'Female', 'Male',
        'Female', 'Male', 'Male', 'Female', 'Male',
        'Female', 'Male', 'Female', 'Male', 'Female'
    ],

    'contract_type': [
        'Permanent', 'Contract', 'Permanent', 'Permanent', 'Contract',
        'Permanent', 'Contract', 'Permanent', 'Contract', 'Permanent',
        'Contract', 'Permanent', 'Permanent', 'Contract', 'Permanent'
    ],

    'experience': [2, 5, 4, 8, 12, 3, 7, 5, 15, 10, 3, 6, 9, 14, 7],

    'salary': [
        25000, 30000, 28000, 50000, 55000,
        27000, 42000, 35000, 60000, 48000,
        26000, 40000, 52000, 65000, 45000
    ],

    'performance_score': [
        7, 8, 6, 9, 8, 7, 9, 8, 9, 8,
        6, 8, 9, 10, 8
    ],

    'bonus': [
        2000, 1500, 1800, 5000, 4500,
        2000, 4000, 3000, 6000, 4500,
        1500, 3500, 5000, 6500, 4000
    ]
}

df = pd.DataFrame(data)

df

# Display first 5 rows
df.head()

# Number of rows and columns
df.shape

# Column names and data types
df.info()

# Check missing values
df.isnull().sum()


numeric_cols = [
    'age',
    'experience',
    'salary',
    'performance_score',
    'bonus'
]

# Calculate summary statistics
summary = df[numeric_cols].agg(
    ['mean', 'median', 'min', 'max', 'std']
)

print(summary.round(2))

department_salary = df.groupby('department').agg(
    average_salary=('salary', 'mean'),
    minimum_salary=('salary', 'min'),
    maximum_salary=('salary', 'max'),
    employee_count=('employee_id', 'count')
)

print(department_salary.round(2))

city_salary = df.groupby('city').agg(
    average_salary=('salary', 'mean'),
    total_salary=('salary', 'sum'),
    employee_count=('employee_id', 'count')
)

print(city_salary.round(2))

gender_salary = df.groupby('gender').agg(
    average_salary=('salary', 'mean'),
    average_bonus=('bonus', 'mean'),
    employee_count=('employee_id', 'count')
)

print(gender_salary.round(2))

contract_salary = df.groupby('contract_type').agg(
    average_salary=('salary', 'mean'),
    average_bonus=('bonus', 'mean'),
    average_experience=('experience', 'mean')
)

print(contract_salary.round(2))

correlation = df[numeric_cols].corr()

print(correlation.round(2))

plt.figure(figsize=(8, 5))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Employee Data Correlation Heatmap')
plt.show()

# Extract upper triangle to avoid duplicate pairs
upper = correlation.where(
    np.triu(np.ones(correlation.shape), k=1).astype(bool)
)

top_3 = upper.stack().abs().sort_values(ascending=False).head(3)

print("Top 3 correlation relationships:")
print(top_3)


# Question 1:
# Which department has the highest average salary?

q1 = df.groupby('department')['salary'].mean()

print("Q1: Department with highest average salary")
print(q1.idxmax(), "₹", round(q1.max(), 2))


# Question 2:
# Which city has the highest average salary?

q2 = df.groupby('city')['salary'].mean()

print("\nQ2: City with highest average salary")
print(q2.idxmax(), "₹", round(q2.max(), 2))


# Question 3:
# Which contract type receives the highest average bonus?

q3 = df.groupby('contract_type')['bonus'].mean()

print("\nQ3: Contract type with highest average bonus")
print(q3.idxmax(), "₹", round(q3.max(), 2))


# Question 4:
# Is experience related to salary?

q4 = df['experience'].corr(df['salary'])

print("\nQ4: Correlation between experience and salary")
print(round(q4, 2))


# Question 5:
# Which department has the highest performance score?

q5 = df.groupby('department')['performance_score'].mean()

print("\nQ5: Department with highest average performance")
print(q5.idxmax(), round(q5.max(), 2))



department_budget = pd.DataFrame({
    'department': ['IT', 'HR', 'Finance', 'Sales'],
    'annual_budget': [1000000, 500000, 800000, 900000]
})

print(department_budget)

merged_df = pd.merge(
    df,
    department_budget,
    on='department',
    how='left'
)

print(merged_df.head())


budget_analysis = merged_df.groupby('department').agg(
    average_salary=('salary', 'mean'),
    annual_budget=('annual_budget', 'first'),
    employee_count=('employee_id', 'count')
)

print(budget_analysis.round(2))


pivot = pd.pivot_table(
    df,
    values='salary',
    index='department',
    columns='gender',
    aggfunc='mean'
)

print(pivot.round(2))


pivot_bonus = pd.pivot_table(
    df,
    values='bonus',
    index='department',
    columns='contract_type',
    aggfunc='mean',
    fill_value=0
)

print(pivot_bonus.round(2))


df.to_csv('employee_data.csv', index=False)

print("Dataset saved successfully!")





