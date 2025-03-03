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
print(df["Month"].value_counts())  # Now it should work



# Simulating non-renewable energy usage (assuming it is a fixed higher value for comparison)
df["Non-Renewable Energy (MW)"] = df["Solar Energy (MW)"] + df["Wind Energy (MW)"] + df["Hydro Energy (MW)"] + 500

# Convert Date column to datetime format
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.month
df["Hour"] = df["Date"].dt.hour if "Hour" in df.columns else 12  # Assuming midday if no hourly data

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

# Analyze peak production times for solar and wind energy
def analyze_peak_production():
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="Month", y="Solar Energy (MW)", data=df)
    plt.title("Solar Energy Production by Month")
    plt.xlabel("Month")
    plt.ylabel("Solar Energy (MW)")
    plt.show()
    
    plt.figure(figsize=(10, 5))
    sns.boxplot(x="Month", y="Wind Energy (MW)", data=df)
    plt.title("Wind Energy Production by Month")
    plt.xlabel("Month")
    plt.ylabel("Wind Energy (MW")
    plt.show()
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(x="Hour", y="Solar Energy (MW)", data=df)
    plt.title("Solar Energy Production by Hour (Peak Analysis)")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Solar Energy (MW)")
    plt.show()
    
    plt.figure(figsize=(10, 5))
    sns.lineplot(x="Hour", y="Wind Energy (MW)", data=df)
    plt.title("Wind Energy Production by Hour (Peak Analysis)")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Wind Energy (MW)")
    plt.show()
    
    # Bar plot for average energy production by month
    plt.figure(figsize=(10, 5))
    df.groupby("Month")[["Solar Energy (MW)", "Wind Energy (MW)", "Hydro Energy (MW)"]].mean().plot(kind='bar', figsize=(12,6))
    plt.title("Average Monthly Energy Production")
    plt.xlabel("Month")
    plt.ylabel("Energy (MW)")
    plt.legend()
    plt.xticks(rotation=0)
    plt.show()
    
    # Histogram plot for energy distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df["Solar Energy (MW)"], bins=30, kde=True, color='orange', label='Solar Energy')
    sns.histplot(df["Wind Energy (MW)"], bins=30, kde=True, color='blue', label='Wind Energy')
    sns.histplot(df["Hydro Energy (MW)"], bins=30, kde=True, color='green', label='Hydro Energy')
    plt.title("Distribution of Energy Production")
    plt.xlabel("Energy (MW)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()

# Train a Linear Regression Model to predict future energy generation
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
train_predictive_model()

