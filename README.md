NAME:shaik sameera INTENSHIP ID:CITS3193 NO:OF:WEEKS:6 weeks PROJECT NAME: Data Analytics PROJECT SCOPE:personal Expense Analytics


#code

Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

Step 2: Create Expense Dataset
data = {
    'Month':['Jan','Feb','Mar','Apr','May','Jun'],
    'Food':[5000,5500,6000,5800,6200,6500],
    'Transport':[2000,2200,2500,2300,2700,3000],
    'Shopping':[3000,3500,3200,4000,4500,5000],
    'Bills':[1500,1500,1600,1700,1800,1900]
}

df = pd.DataFrame(data)

df

Step 3: Data Cleaning
print("Missing Values")

print(df.isnull().sum())

df.drop_duplicates(inplace=True)

print("Data Cleaning Completed")

Step 4: Total Monthly Expense
df['Total_Expense'] = (
    df['Food']
    + df['Transport']
    + df['Shopping']
    + df['Bills']
)

df


Step 5: Expense Trend Visualization
plt.figure(figsize=(8,5))

plt.plot(
    df['Month'],
    df['Total_Expense'],
    marker='o'
)

plt.title("Monthly Expense Trend")
plt.xlabel("Month")
plt.ylabel("Expense")

plt.grid(True)

plt.show()
Step 6: Category-wise Expense Analysis
categories = [
    'Food',
    'Transport',
    'Shopping',
    'Bills'
]

totals = [
    df['Food'].sum(),
    df['Transport'].sum(),
    df['Shopping'].sum(),
    df['Bills'].sum()
]


plt.figure(figsize=(8,5))

plt.bar(categories, totals)

plt.title("Expense Category Analysis")
plt.xlabel("Category")
plt.ylabel("Amount")

plt.show()

7: Expense Distribution Pie Chart
plt.figure(figsize=(6,6))

plt.pie(
    totals,
    labels=categories,
    autopct='%1.1f%%'
)

plt.title("Expense Distribution")

plt.show()

Step 8: Analysis Commands
print("Total Expense =", df['Total_Expense'].sum())

print("Average Monthly Expense =", df['Total_Expense'].mean())

print("Highest Expense =", df['Total_Expense'].max())

print("Lowest Expense =", df['Total_Expense'].min())



Step 9: Prediction Model

Predict next month's expense.

X = np.array([1,2,3,4,5,6]).reshape(-1,1)

y = np.array(df['Total_Expense'])

model = LinearRegression()

model.fit(X,y)

prediction = model.predict([[7]])

print("Predicted Expense for Next Month =", prediction[0])


Step 10: Forecast Graph
future_months = [1,2,3,4,5,6,7]

future_expense = list(y)

future_expense.append(prediction[0])

plt.figure(figsize=(8,5))

plt.plot(
    future_months,
    future_expense,
    marker='o'
)

plt.title("Expense Forecast")
plt.xlabel("Month Number")
plt.ylabel("Expense")

plt.grid(True)

plt.show()


Step 11: Dashboard Creation
fig, axs = plt.subplots(2,2, figsize=(12,8))

axs[0,0].plot(df['Month'], df['Total_Expense'])
axs[0,0].set_title('Total Expense')

axs[0,1].bar(df['Month'], df['Food'])
axs[0,1].set_title('Food')

axs[1,0].bar(df['Month'], df['Shopping'])
axs[1,0].set_title('Shopping')

axs[1,1].bar(df['Month'], df['Transport'])
axs[1,1].set_title('Transport')

plt.tight_layout()

plt
