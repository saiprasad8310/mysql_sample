import matplotlib.pyplot as plt
import seaborn as sns  

plt.figure(figsize=(12,6))  
sns.lineplot(data=df, x="Year", y="Consumption_MWh", hue="Energy_Type")  
plt.title("Renewable vs. Non-Renewable Energy Usage Over Time")  
plt.show()


