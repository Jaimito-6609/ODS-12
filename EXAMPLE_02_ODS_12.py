# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:19:29 2024

@author: Jaime
"""
"""
Artificial intelligence (AI) has become crucial in accounting, improving 
efficiency and decision making according to Peng et al. This contributes to 
several Sustainable Development Goals (SDGs), such as decent work, economic 
growth and innovation. They highlight how the integration of AI in companies, 
development of algorithms for global challenges and internal transformation, 
play a vital role in sustainable development. These areas show the 
transformative impact of AI on accounting processes and business sustainability 
Peng et al. (2023). Peng, Y., Ahmad, S. F., Ahmad, A. B., Shaikh, M., Daoud, 
M. K., & Alhamdi, F. M. H. (2023). Riding the Waves of Artificial Intelligence 
in Advancing Accounting and Its Implications for Sustainable Development 
Goals. Sustainability, 15(19), 14165. https://doi.org/10.3390/su151914165.

The provided Python code implements an algorithm to analyze the impact of AI 
integration on accounting processes and business sustainability, contributing 
to Sustainable Development Goals (SDGs). The dataset, which includes accounting 
indicators influenced by AI, is processed to select key features and define 
the target variable, which is the sustainability score. The features are 
normalized, and the data is split into training and testing sets. A 
RandomForestRegressor model is trained on the training data and evaluated on 
the test data. The model, along with the scaler, is saved for future use. 
Additionally, a function is defined to generate insights and recommendations 
for integrating AI in accounting practices, focusing on enhancing efficiency, 
decision quality, innovation level, and economic growth. The generated 
insights are saved for further use.
"""
#==============================================================================
# EXAMPEL 02 ODS 12
#==============================================================================
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset
# The dataset should contain accounting indicators influenced by AI integration in companies
# Example: dataset = pd.read_csv('accounting_data.csv')

# Placeholder for the dataset path
dataset_path = 'accounting_data.csv'
dataset = pd.read_csv(dataset_path)

# Feature selection and target definition
# Comment: Select relevant features for modeling and define the target variable
features = ['AI_integration_level', 'efficiency_improvement', 'decision_quality', 'economic_growth_impact', 'innovation_level']
target = 'sustainability_score'

X = dataset[features]
y = dataset[target]

# Preprocessing the data
# Comment: Normalize the features to ensure consistent scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Save the model and scaler for future use
joblib.dump(model, 'accounting_ai_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Generate insights for AI-driven accounting improvements
# Comment: Define a function to generate insights and recommendations for integrating AI in accounting practices
def generate_insights(company_data):
    insights = []
    for index, row in company_data.iterrows():
        insight = f"Company {row['company_name']}: Increase AI integration to enhance efficiency, decision quality, and innovation level. Focus on leveraging AI for economic growth and sustainability improvements."
        insights.append(insight)
    return insights

# Example usage of insight generation
insights = generate_insights(dataset)
for insight in insights[:5]:
    print(insight)

# Save the generated insights
with open('ai_accounting_insights.txt', 'w') as f:
    for insight in insights:
        f.write(f"{insight}\n")
