Patient Survival Prediction using Logistic Regression

Overview

This project applies logistic regression to predict patient survival based on age. The dataset consists of 167 patients, with:
Status (0 = Alive, 1 = Deceased)
Age (in years)


We use Scikit-Learn's logistic regression to model the probability of patient survival as a function of age and visualize the logistic regression curve.

Dataset
The dataset is stored in patient1.csv, which contains:
status: Patient survival status (0 = alive, 1 = deceased)
age: Patient's age in years

Example of the dataset:

status	age
0	27
0	59
0	77
0	54
1	87

Objective:

Train a logistic regression model using age as the input and status as the output.
Determine the relationship between age and survival probability.
Plot the logistic regression curve to visualize the trend.
Installation & Setup

Ensure you have Python and the necessary dependencies installed. You can install them using:
pip install numpy pandas matplotlib seaborn scikit-learn

How to Run the Script
Clone the repository:

git clone https://github.com/your-username/patient-survival-logistic-regression.git
cd patient-survival-logistic-regression

Ensure the dataset patient1.csv is in the same directory as the script.
Run the logistic regression analysis script:

python logistic_regression.py
Python Script (logistic_regression.py)
python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("patient1.csv")

# Define input (age) and output (status)
X = data[['age']].values  # Independent variable: Age
y = data['status'].values  # Dependent variable: Survival status

# Train logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X, y)

# Extract coefficients
coef = log_reg.coef_[0][0]
intercept = log_reg.intercept_[0]

# Generate age values for plotting
age_range = np.linspace(data['age'].min(), data['age'].max(), 100).reshape(-1, 1)

# Compute predicted probabilities
probabilities = log_reg.predict_proba(age_range)[:, 1]

# Plot logistic regression curve
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['age'], y=data['status'], alpha=0.6, label="Actual Data")
plt.plot(age_range, probabilities, color='red', linewidth=2, label="Logistic Regression Curve")
plt.xlabel("Age")
plt.ylabel("Probability of Death (Status = 1)")
plt.title("Logistic Regression Curve: Age vs. Patient Status")
plt.legend()
plt.grid(True)
plt.show()

# Print final weights
print(f"Coefficient (Weight for Age): {coef:.4f}")
print(f"Intercept: {intercept:.4f}")

Expected Output

1. Logistic Regression Weights
After running the script, it will print:
Coefficient (Weight for Age): 0.0418
Intercept: -4.2449
The coefficient (0.0418) indicates that as age increases, the probability of death increases.
The intercept (-4.2449) shifts the curve appropriately.

3. Logistic Regression Curve
The script will generate the following graph:
X-axis: Age of patients.
Y-axis: Probability of death.
Blue dots: Actual patient data.
Red curve: Predicted probability from logistic regression.
Interpretation of Results
Younger patients (20-40 years old) have a very low probability of death.
Older patients (70+ years old) have an increased probability of death.
Ages 50-60 show a transition where some patients survive while others do not.
The logistic function follows an S-curve, meaning the probability gradually increases as age rises.
