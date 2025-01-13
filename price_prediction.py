#!/usr/bin/env python
# coding: utf-8

# ## Airbnb Analytics: Predicting Optimal Pricing for Market Success

# ### Import Required Packages and libraries

# In[3]:


import numpy as np 
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pylab as plt

import dmba
from dmba import plotDecisionTree, classificationSummary, regressionSummary
from dmba import regressionSummary, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score
from sklearn import preprocessing
from sklearn.metrics import pairwise
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler 

import dmba
import folium


# ### Load the dataset

# In[6]:


df = pd.read_csv('/Users/sindhujaarivukkarasu/Documents/BAN 620 Data Mining/project-620/AB_US_2023.csv')
df.head() 


# ### Dataset pre-processing

# In[9]:


df.info()


# ### 1. Missing Value Treatment

# In[12]:


#Checking Missing values
pd.DataFrame(df).isna().sum()


# In[14]:


import seaborn as sns
sns.heatmap(df.isnull(), cbar=False)
plt.show()


# ### Dropping columns with high NA and ID column

# In[16]:


df.drop(columns=['id','last_review', 'reviews_per_month', 'neighbourhood_group'], inplace=True)


# In[19]:


summary_stats = np.round(df.describe(), decimals=4)
print(summary_stats)


# In[21]:


#price boxplot- detecting outliers
#converting boxplot from normal to log scale
#change subplots rows and columns as required
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))

# regular scale
ax = df.boxplot(column='minimum_nights', ax=axes[0])
ax.set_xlabel('minimum_nights')
ax.set_ylabel('minimum_nights')

# log scale
ax = df.boxplot(column='minimum_nights', ax=axes[1])
ax.set_xlabel('log minimum_nights')
ax.set_ylabel('log minimum_nights')
ax.set_yscale('log')

# suppress the title
axes[0].get_figure().suptitle('')
plt.tight_layout()

plt.show()


# ### 2. Removing Outliers for Price Variable

# In[24]:


lower_bound = .05
upper_bound = .97
airbnb_df = df[df['price'].between(df['price'].quantile(lower_bound), df['price'].quantile(upper_bound))]


# In[26]:


summary_stats = np.round(airbnb_df.describe(), decimals=4)
print(summary_stats)


# In[28]:


row_count = airbnb_df.shape[0]
print("The number of rows in the DataFrame is:", row_count)


# ### Exploratory Data Analysis

# In[31]:


# Bar chart for room_type distribution across cities
room_type_counts = airbnb_df['room_type'].value_counts()
room_type_counts.plot(kind='bar', color='skyblue', edgecolor='black')

# Adding labels and title
plt.xlabel('Room Type')
plt.ylabel('Number of Listings')
plt.title('Room Type Distribution across cities')

# Show the plot
plt.show()


# In[33]:


# Room_type and median prices
airbnb_df.groupby('room_type')['price'].median()


# In[35]:


# Bar chart for number of listings in each city
city_counts = airbnb_df['city'].value_counts()

# Plotting the bar chart
city_counts.plot(kind='bar', color='skyblue', edgecolor='black')

# Adding labels and title
plt.xlabel('City')
plt.ylabel('Number of Listings')
plt.title('Number of Listings in Each City')

# Show the plot
plt.show()


# In[37]:


# Bar chart for average price across each city
average_price_by_city = airbnb_df.groupby('city')['price'].mean().reset_index()

# Plotting the bar chart
plt.figure(figsize=(10, 6))  # Adjust figure size if needed
plt.bar(average_price_by_city['city'], average_price_by_city['price'], color='skyblue', edgecolor='black')

# Adding labels and title
plt.xlabel('City')
plt.ylabel('Average Price')
plt.title('Average Price Across Each City')

# Rotating x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()


# In[39]:


# Boxplot for distribution of listing availability across cities
plt.figure(figsize=(10, 8))  # Adjust figure size if needed
sns.boxplot(x='city', y='availability_365', data=airbnb_df, palette='pastel')

# Adding labels and title
plt.xlabel('City')
plt.ylabel('Listing Availability (in days)')
plt.title('Distribution of Listing Availability Across Cities')

# Rotating x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.show()


# In[41]:


# Stacked Bar chart for room_type vs city
x_variable = 'city'
y_variable = 'room_type'

# Create a pivot table for the stacked bar chart
pivot_data = airbnb_df.pivot_table(index=x_variable, columns=y_variable, aggfunc='size', fill_value=0)

# Normalize the values to percentages
pivot_data_percentage = pivot_data.div(pivot_data.sum(axis=1), axis=0) * 100

# Plotting a stacked bar chart with default colors
plt.figure(figsize=(12, 6))
pivot_data_percentage.plot(kind='bar', stacked=True)
plt.title(f'Stacked Bar Chart (Percentage): {y_variable} vs {x_variable}')
plt.xlabel(x_variable)
plt.ylabel('Percentage')
plt.xticks(rotation=45, ha='right')
plt.legend(title=y_variable, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[43]:


# Heatmap for price vs numerical variables
selected_columns = ['price', 'availability_365', 'calculated_host_listings_count', 'minimum_nights', 'number_of_reviews', 'number_of_reviews_ltm']
selected_data = airbnb_df[selected_columns]

# Calculate the correlation matrix
correlation_matrix = selected_data.corr()

# Set up the matplotlib figure
plt.figure(figsize=(5, 5))

# Create a heatmap using Seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

# Set the title of the heatmap
plt.title('Correlation Heatmap: Price vs Numerical Variables')

# Show the plot
plt.show()


# In[45]:


MapModel = airbnb_df[['latitude', 'longitude']]
model = KMeans()
kmeans = KMeans(n_clusters = 15, random_state=42,n_init=10).fit(MapModel)
kmeans.cluster_centers_
cluster_map = folium.Map([41.8781, -87.6298], zoom_start=4)
for i in range(kmeans.cluster_centers_.shape[0]):
    num = sum(kmeans.labels_ == i)
    folium.CircleMarker([kmeans.cluster_centers_[i,0], kmeans.cluster_centers_[i,1]],
                        radius=15,
                        popup=str(num) + ' Listings Associated with this Cluster',
                        fill_color="#3db7e4",
                        ).add_to(cluster_map)
cluster_map


# In[47]:


### Price and reviews:
plt.figure(figsize=(4,4))
sns.scatterplot(x='price',y='number_of_reviews',data=airbnb_df)
plt.title("Relationship between price and number of reviews",fontsize=15)
plt.xlabel("Price",fontsize=12)
plt.ylabel("Number of review",fontsize=12)
plt.show()



# ### Model 1: Price prediction

# ### Creating Dummies for Categorical variables - Room_Type, City

# In[51]:


airbnb_df = pd.get_dummies(airbnb_df, columns=['room_type','city'], prefix=['room_type','city'], drop_first=True) 


# In[53]:


airbnb_df.columns


# ### Multiple Linear Regression for Price Prediction - Base Model

# In[56]:


predictors = ['minimum_nights', 'number_of_reviews',
       'calculated_host_listings_count', 'availability_365',
       'number_of_reviews_ltm', 'room_type_Hotel room',
       'room_type_Private room', 'room_type_Shared room', 'city_Austin',
       'city_Boston', 'city_Broward County', 'city_Cambridge', 'city_Chicago',
       'city_Clark County', 'city_Columbus', 'city_Denver', 'city_Jersey City',
       'city_Los Angeles', 'city_Nashville', 'city_New Orleans',
       'city_New York City', 'city_Oakland', 'city_Pacific Grove',
       'city_Portland', 'city_Rhode Island', 'city_Salem', 'city_San Diego',
       'city_San Francisco', 'city_San Mateo County',
       'city_Santa Clara County', 'city_Santa Cruz County', 'city_Seattle',
       'city_Twin Cities MSA', 'city_Washington D.C.']

outcome = 'price'

# partition data
X = airbnb_df[predictors]
y = airbnb_df[outcome]
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)
train_X.head()
air_lm = LinearRegression()
air_lm.fit(train_X, train_y)


# ### Measuring the performance on Validation Set

# In[59]:


# Use predict() to make predictions on a new set
air_lm_pred = air_lm.predict(valid_X)

result = pd.DataFrame({'Predicted': air_lm_pred, 'Actual': valid_y,
                       'Residual': valid_y - air_lm_pred})
print(result.head(20))

# Compute common accuracy measures
regressionSummary(valid_y, air_lm_pred)


# ### Random Forest

# In[62]:


rf_df = pd.read_csv('/Users/sindhujaarivukkarasu/Documents/BAN 620 Data Mining/project-620/AB_US_2023.csv', header = 0)


# In[64]:


rf_df.drop(columns=['id','last_review', 'reviews_per_month', 'neighbourhood_group'], inplace=True)


# In[66]:


# Removing Outliers for Price
lower_bound = .05
upper_bound = .97
rf1_df = df[rf_df['price'].between(df['price'].quantile(lower_bound), rf_df['price'].quantile(upper_bound))]


# In[68]:


rf1_df = pd.get_dummies(rf1_df, columns=['room_type','city'], prefix=['room_type','city'], drop_first=False)
rf1_df.columns


# In[70]:


predictors = ['minimum_nights','number_of_reviews','calculated_host_listings_count',
             'availability_365','number_of_reviews_ltm', 'room_type_Entire home/apt',
             'room_type_Hotel room', 'room_type_Private room',
       'room_type_Shared room', 'city_Asheville', 'city_Austin', 'city_Boston',
       'city_Broward County', 'city_Cambridge', 'city_Chicago',
       'city_Clark County', 'city_Columbus', 'city_Denver', 'city_Jersey City',
       'city_Los Angeles', 'city_Nashville', 'city_New Orleans',
       'city_New York City', 'city_Oakland', 'city_Pacific Grove',
       'city_Portland', 'city_Rhode Island', 'city_Salem', 'city_San Diego',
       'city_San Francisco', 'city_San Mateo County',
       'city_Santa Clara County', 'city_Santa Cruz County', 'city_Seattle',
       'city_Twin Cities MSA', 'city_Washington D.C.']
outcome = 'price'

# partition data
A = rf1_df[predictors]
b = rf1_df[outcome]
train_A, valid_a, train_b, valid_b = train_test_split(A, b, test_size=0.4, random_state=1)


# In[72]:


rf = RandomForestRegressor(n_estimators=50, random_state=1)
rf.fit(train_A, train_b)


# ### Variable Importance Plot

# In[75]:


importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

variables_df = pd.DataFrame({'feature': train_A.columns, 'importance': importances, 'std': std})
variables_df = variables_df.sort_values('importance')
print(variables_df)
variables_df.to_csv('variables_list.csv', index=True)

ax = variables_df.plot(kind='barh', xerr='std', x='feature', legend=False)
ax.set_ylabel('')

plt.tight_layout()
plt.show()


# ### Backward elimination

# In[78]:


def train_model(variables):
    model = LinearRegression()
    model.fit(train_X[variables], train_y)
    return model

def score_model(model, variables):
    return AIC_score(train_y, model.predict(train_X[variables]), model)

bestBE_model, best_variables = backward_elimination(train_X.columns, train_model, score_model, verbose=True)

print(best_variables)


# ### Final Model using best predictor variables

# In[81]:


predictors = ['availability_365','calculated_host_listings_count','minimum_nights','number_of_reviews','room_type_Private room'] #n=5, RMSE-140.1425

outcome = 'price'

# partition data
X = airbnb_df[predictors]
y = airbnb_df[outcome]
train_X, valid_X, train_y, valid_y = train_test_split(X, y, test_size=0.4, random_state=1)
air_lm_final = LinearRegression()
air_lm_final.fit(train_X, train_y)

# print coefficients
print('intercept ', air_lm_final.intercept_)
print(pd.DataFrame({'Predictor': X.columns, 'coefficient': air_lm_final.coef_}))


# In[83]:


# Use predict() to make predictions on a new set
pred = air_lm_final.predict(valid_X)

result = pd.DataFrame({'Predicted': pred, 'Actual': valid_y,
                       'Residual': valid_y - pred})
print(result.head(20))

# Compute common accuracy measures
regressionSummary(valid_y, pred)


# ### Plotting the residuals

# In[86]:


pred = air_lm_final.predict(valid_X)
all_residuals = valid_y - pred

# Determine the percentage of datapoints with a residual in [-300, 300] = approx. 95\%
print(len(all_residuals[(all_residuals > -300) & (all_residuals < 300)]) / len(all_residuals))

ax = pd.DataFrame({'Residuals': all_residuals}).hist(bins=25)

plt.tight_layout()
plt.show()


# ### Save the final model

# In[89]:


import pickle

# Save the final linear regression model to a file
with open('price_prediction_model.pkl', 'wb') as f:
    pickle.dump(air_lm_final, f)

print("Model saved as 'price_prediction_model.pkl'")

