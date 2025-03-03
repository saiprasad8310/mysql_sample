import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the cleaned renewable energy dataset
df = pd.read_csv("Cleaned_Renewable_Energy_Production.csv")

# Display basic dataset information
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isna().sum())
print("\nNull Values:")
print(df.isnull().sum())

print("Columns in DataFrame:", df.columns)  # Check if "Date" exists
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")  # Convert to datetime
df["Month"] = df["Date"].dt.month  # Extract month
df["Hour"] = df["Date"].dt.hour if "Hour" in df.columns else 12  # Assuming midday if no hourly data

# Simulating non-renewable energy usage
df["Non-Renewable Energy (MW)"] = df["Solar Energy (MW)"] + df["Wind Energy (MW)"] + df["Hydro Energy (MW)"] + 500

# Drop rows with missing values
df.dropna(inplace=True)

# Group by Month and summarize energy production
print("\nAverage Energy Production by Month:")
print(df.groupby("Month")[['Solar Energy (MW)', 'Wind Energy (MW)', 'Hydro Energy (MW)']].mean())

def plot_energy_comparison():
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df["Date"], y=df["Solar Energy (MW)"], label="Solar Energy")
    sns.lineplot(x=df["Date"], y=df["Wind Energy (MW)"], label="Wind Energy")
    sns.lineplot(x=df["Date"], y=df["Hydro Energy (MW)"], label="Hydro Energy")
    sns.lineplot(x=df["Date"], y=df["Non-Renewable Energy (MW)"], label="Non-Renewable Energy", linestyle='dashed')
    plt.title("Renewable vs Non-Renewable Energy Usage Over Time")
    plt.xlabel("Date")
    plt.ylabel("Energy (MW)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

def analyze_peak_production():
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    sns.boxplot(x="Month", y="Solar Energy (MW)", data=df, ax=axes[0, 0])
    axes[0, 0].set_title("Solar Energy Production by Month")
    
    sns.boxplot(x="Month", y="Wind Energy (MW)", data=df, ax=axes[0, 1])
    axes[0, 1].set_title("Wind Energy Production by Month")
    
    df.groupby("Month")[["Solar Energy (MW)", "Wind Energy (MW)", "Hydro Energy (MW)"]].mean().plot(kind='bar', ax=axes[1, 0])
    axes[1, 0].set_title("Average Monthly Energy Production")
    axes[1, 0].set_xlabel("Month")
    axes[1, 0].set_ylabel("Energy (MW)")
    
    sns.heatmap(df[["Solar Energy (MW)", "Wind Energy (MW)", "Hydro Energy (MW)", "Non-Renewable Energy (MW)"]].corr(), annot=True, cmap="coolwarm", ax=axes[1, 1])
    axes[1, 1].set_title("Correlation Heatmap")
    
    plt.tight_layout()
    plt.show()

def energy_distribution_pie_chart():
    energy_totals = df[["Solar Energy (MW)", "Wind Energy (MW)", "Hydro Energy (MW)", "Non-Renewable Energy (MW)"]].sum()
    labels = ["Solar", "Wind", "Hydro", "Non-Renewable"]
    colors = ["gold", "skyblue", "green", "red"]
    
    plt.figure(figsize=(8, 8))
    plt.pie(energy_totals, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title("Energy Distribution")
    plt.show()

def train_predictive_model():
    features = ["Month", "Hour"]
    target = "Solar Energy (MW)"
    
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
    
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed')
    plt.xlabel("Actual Solar Energy (MW)")
    plt.ylabel("Predicted Solar Energy (MW)")
    plt.title("Actual vs Predicted Solar Energy")
    plt.show()

# Run analysis
plot_energy_comparison()
analyze_peak_production()
energy_distribution_pie_chart()
train_predictive_model()
