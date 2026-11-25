import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Libraries imported successfully!")

np.random.seed(42)

n = 1470

data = {
    "customer_id": range(1, n + 1),

    "name": np.random.choice(
        ["john smith", "MARY JONES", "David Brown",
         "  alice green ", "ROBERT WHITE"], n
    ),

    "age": np.random.randint(21, 65, n).astype(str),

    "city": np.random.choice(
        ["chennai", "COIMBATORE", "Madurai",
         "  trichy ", "SALEM"], n
    ),

    "salary": np.random.randint(25000, 90000, n).astype(str),

    "join_date": pd.date_range(
        start="2020-01-01", periods=n, freq="D"
    ).strftime("%Y-%m-%d"),

    "department": np.random.choice(
        ["IT", "HR", "Finance", "sales"], n
    )
}

df = pd.DataFrame(data)

# Introduce salary outliers
df.loc[100:109, "salary"] = "250000"

# Introduce missing values (214 total)
df.loc[20:69, "age"] = np.nan
df.loc[70:129, "salary"] = np.nan
df.loc[130:169, "city"] = np.nan
df.loc[170:199, "department"] = np.nan
df.loc[200:233, "join_date"] = np.nan

# Add 10 duplicate records
df = pd.concat([df, df.iloc[:10]], ignore_index=True)

print(df.head())
print("\nDataset shape:", df.shape)

print(df.head())
print(df.shape)
print(df.info())

print(df.isnull().sum())
print("Total missing values:", df.isnull().sum().sum())

df["age"] = pd.to_numeric(df["age"])

df["salary"] = pd.to_numeric(df["salary"])

df["join_date"] = pd.to_datetime(df["join_date"])
df["age"] = df["age"].fillna(df["age"].median())

df["salary"] = df["salary"].fillna(df["salary"].median())
df["city"] = df["city"].fillna(df["city"].mode()[0])

df["department"] = df["department"].fillna(
    df["department"].mode()[0]
)
df["join_date"] = df["join_date"].fillna(
    df["join_date"].median()
)
print(df.isnull().sum())
print("Total missing:", df.isnull().sum().sum())

print(df.dtypes)

df["customer_id"] = df["customer_id"].astype(int)

df["age"] = df["age"].astype(int)

df["salary"] = df["salary"].astype(float)

df["join_date"] = pd.to_datetime(df["join_date"])

df["city"] = df["city"].astype("category")

df["department"] = df["department"].astype("category")

print(df.dtypes)

print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

print("After removing duplicates:", df.shape)

print("Remaining duplicates:", df.duplicated().sum())
df["name"] = df["name"].str.strip().str.title()

df["city"] = (
    df["city"]
    .astype(str)
    .str.strip()
    .str.title()
)
df["department"] = (
    df["department"]
    .astype(str)
    .str.strip()
    .str.title()
)
print(df[["name", "city", "department"]].head(10))

Q1 = df["salary"].quantile(0.25)

Q3 = df["salary"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR

upper = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)

print("Lower limit:", lower)
print("Upper limit:", upper)

outliers = df[
    (df["salary"] < lower) |
    (df["salary"] > upper)
]

print("Number of outliers:", len(outliers))

plt.figure(figsize=(8, 4))

plt.boxplot(df["salary"])

plt.title("Salary Outlier Detection")

plt.ylabel("Salary")

plt.show()

df["salary"] = df["salary"].clip(
    lower=lower,
    upper=upper
)

print("Outliers handled successfully!")

# Original dataset
original_rows = 1480
original_missing = 214
original_duplicates = 10

# Cleaned dataset
clean_rows = len(df)
clean_missing = df.isnull().sum().sum()
clean_duplicates = df.duplicated().sum()

summary = pd.DataFrame({
    "Metric": [
        "Rows",
        "Missing Values",
        "Duplicates"
    ],

    "Before Cleaning": [
        original_rows,
        original_missing,
        original_duplicates
    ],

    "After Cleaning": [
        clean_rows,
        clean_missing,
        clean_duplicates
    ]
})

print(summary)

df.to_csv("clean_data.csv", index=False)

print("Cleaned dataset saved successfully!")

cleaned_data = pd.read_csv("clean_data.csv")

print(cleaned_data.head())

print("Final shape:", cleaned_data.shape)