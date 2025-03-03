import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings as wn

#df = pd.read_csv("time.csv")

"""print(df.head())
df.shape
df.info()
df.describe()
df.columns.tolist()
df.isnull().sum()
df.nunique()"""
"""quality_counts = df['12-Hour'].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(quality_counts.index, quality_counts, color='black')
plt.title('Count Plot of Quality')
plt.xlabel('12-Hour')
plt.ylabel('24-Hour')
plt.show()

print(df.columns.tolist())"""



"""# Set the style for the plots
sns.set_style("darkgrid")

# Select numerical columns from the dataframe
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns

# Set the size of the figure based on the number of numerical columns
plt.figure(figsize=(14, len(numerical_columns) * 3))

# Loop through each numerical feature and create a histogram with KDE
for idx, feature in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 2, idx)
    sns.histplot(df[feature], kde=True)
    plt.title(f"{feature} | Skewness: {round(df[feature].skew(), 2)}")

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()"""
#plt.figure(figsize=(10, 8))

# Using Seaborn to create a swarm plot
"""sns.swarmplot(x="quality", y="alcohol", data=df, palette='viridis')

plt.title('Swarm Plot for Quality and Alcohol')
plt.xlabel('Quality')
plt.ylabel('Alcohol')
plt.show()"""

df = pd.read_csv("https://raw.githubusercontent.com/siglimumuni/Datasets/master/customer-data.csv")
#df.shape
#df.head()
#df.info()
#df.isna().sum()
#df.groupby(by="income")["credit_score"].mean()
#print(df.isna().sum())
#df.groupby(by="income")["credit_score"].mean()
#print(df.groupby(by="income")["credit_score"].mean())
def impute_creditscore(income_classes):
    """This function takes a list of income groups and imputes the missing values of each based on the mean credit score for          each group"""
    #iterate through each income group
    for income_class in income_classes:      
        
        #create a subset of dataframe to use as filter
        mask = df["income"] == income_class
        
        #calculate the mean for the income group
        mean = df[df["income"] == income_class]["credit_score"].mean()
        
        #fill the missing values with mean of credit score for group
        df.loc[mask,"credit_score"] = df.loc[mask,'credit_score'].fillna(mean)
#Apply the function to the dataframe
income_groups = ["poverty","upper class","middle class","working class"]
impute_creditscore(income_groups)

#check for missing values
df.isnull().sum()
#print(df)
df.groupby(by="driving_experience")["annual_mileage"].mean()    
#print(df.groupby(by="driving_experience")["annual_mileage"].mean())
#Calculate mean for annual_mileage column
mean_mileage = df["annual_mileage"].mean()

#Fill in null values using the column mean
df["annual_mileage"].fillna(mean_mileage,inplace=True)

#Check for null values
df.isna().sum()
#print(df.isna().sum())
#Delete the id and postal_code columns
#df.drop(["id","postal_code"],axis=1,inplace=True)
#print(df.drop(["id","postal_code"],axis=1,inplace=True))
df["gender"].value_counts()
#print(df["gender"].value_counts())
#sns.countplot(data=df,x="gender")
#plt.title("Number of Clients per Gender")
#plt.ylabel("Number of Clients")
#plt.show()
#Define plot size
plt.figure(figsize=[6,6])

#Define column to use
data = df["income"].value_counts(normalize=True)

#Define labels
labels = ["upper class","middle class","poverty","working class"]

#Define color palette
colors = sns.color_palette('pastel')

#Create pie chart
#plt.pie(data,labels=labels,colors=colors, autopct='%.0f%%')
#lt.title("Proportion of Clients by Income Group")
#plt.show()
#plt.figure(figsize=[8,5])
#sns.countplot(data=df,x="education",order=["university","high school","none"],color="orange")
#plt.title("Number of Clients per Education Level")
#plt.show()

#df["credit_score"].describe()
#print(df["credit_score"].describe())

#plt.figure(figsize=[8,5])
#sns.histplot(data=df,x="credit_score",bins=40).set(title="Distribution of credit scores",ylabel="Number of clients")
#plt.show()

#plt.figure(figsize=[8,5])
#sns.histplot(data=df,x="annual_mileage",bins=20,kde=True).set(title="Distribution of Annual Mileage",ylabel="Number of clients")
#plt.show()
#plt.figure(figsize=[8,5])
#plt.scatter(data=df,x="annual_mileage",y="speeding_violations")
#plt.title("Annual Mileage vrs Speeding Violations")
##plt.xlabel("Annual Mileage")3
#plt.show()
#corr_matrix = df[["speeding_violations","DUIs","past_accidents"]].corr()
#print(corr_matrix)

#plt.figure(figsize=[8,5])
##sns.heatmap(correlation_matrix,annot=True,cmap='Reds')
#plt.title("Correlation between Selected Variables")
#plt.show()

df.groupby('outcome')['annual_mileage'].mean()
print(df.groupby('outcome')['annual_mileage'].mean())
sns.boxplot(data=df,x='outcome', y='annual_mileage')
plt.title("Distribution of Annual Mileage per Outcome")
plt.show()
sns.histplot(df,x="credit_score",hue="outcome",element="step",stat="density")
plt.title("Distribution of Credit Score per Outcome")
plt.show()
#Create a new "claim rate" column
df['claim_rate'] = np.where(df['outcome']==True,1,0)
df['claim_rate'].value_counts()
print(df['claim_rate'].value_counts())

plt.figure(figsize=[8,5])
df.groupby('age')['claim_rate'].mean().plot(kind="bar")
plt.title("Claim Rate by Age Group")
plt.show()
edu_income = pd.pivot_table(data=df,index='education',columns='income',values='claim_rate',aggfunc='mean')
print(edu_income)
fig, axes = plt.subplots(1,2,figsize=(12,4))

#Plot two probability graphs for education and income
for i,col in enumerate(["education","income"]):
    sns.histplot(df, ax=axes[i],x=col, hue="outcome",stat="probability", multiple="fill", shrink=.8,alpha=0.7)
    axes[i].set(title="Claim Probability by "+ col,ylabel=" ",xlabel=" ")
plt.show()

#Create a heatmap to visualize income, education and claim rate
plt.figure(figsize=[8,5])
sns.heatmap(edu_income,annot=True,cmap='coolwarm',center=0.117)
plt.title("Education Level and Income Class")
plt.show()

driv_married = pd.pivot_table(data=df,index='driving_experience',columns='married',values='claim_rate')
plt.figure(figsize=[8,5])
sns.heatmap(driv_married,annot=True,cmap='coolwarm', center=0.117)
plt.title("Driving Experience and Marital Status")
plt.show()

gender_children = pd.pivot_table(data=df,index='gender',columns='children',values='claim_rate')

#Create a heatmap to visualize gender, family status and claim rate
plt.figure(figsize=[8,5])
sns.heatmap(gender_children,annot=True,cmap='coolwarm', center=0.117)
plt.title("Gender and Family Status")
plt.show()