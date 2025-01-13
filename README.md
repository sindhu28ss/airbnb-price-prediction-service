# Airbnb-Price-Prediction
## Background:
Airbnb is a platform that connects travelers with hosts offering accommodations for various needs such as tourism, work trips, weekend getaways, family vacations, or longer stays. Founded in 2008 by two designers, Airbnb has become a globally recognized brand, providing unique travel experiences.
The platform generates revenue by charging fees to both guests and hosts. To remain competitive, Airbnb must continually optimize its pricing strategies. Given the rising demand and supply, predicting pricing factors, user preferences, and market trends becomes increasingly challenging.
Data analysis of Airbnb listings serves as a cornerstone for informed business decisions. It enhances security, deepens understanding of customer and provider behavior, and helps improve Airbnb listing performance. These insights contribute to better guest experiences and increased profitability for the platform.

## Objective:
The objective of this project is to develop a predictive modeling solution that analyzes historical Airbnb listing data to provide actionable insights for:

**Effective pricing optimization:** Empower hosts to set competitive prices based on market trends and listing features.

**Enhanced user satisfaction:** Ensure guests receive value-driven accommodation options.

**Business growth:** Support Airbnb in making data-driven decisions for marketing, profit maximization, and additional services.

## Expected Outcomes:
* A predictive model that estimates Airbnb listing prices based on key features such as availability, room type, and reviews.
* Deployment of the model using Flask, providing a simple and accessible web service.
* Insights that support Airbnb hosts in refining pricing strategies and improving the overall user experience.

## Dataset:
The dataset used for this project can be found on [Kaggle](https://www.kaggle.com/datasets/kritikseth/us-airbnb-open-data/code?datasetId=938452&sortBy=commentCount).
It consists of 232,147 rows and 18 columns, each described below:

| Column Name                     | Description                                                                                     | Data Type          |
|---------------------------------|-------------------------------------------------------------------------------------------------|--------------------|
| **id**                          | Listing Id                                                                                     | Numeric            |
| **name**                        | Listing name                                                                                   | Categorical        |
| **host_id**                     | Id of Host                                                                                     | Numeric            |
| **host_name**                   | Host name                                                                                      | Categorical        |
| **neighbourhood_group**         | Broader category of neighbourhoods where the listing is situated                               | Categorical        |
| **neighbourhood**               | Community where the listing is situated in a city/town                                         | Categorical        |
| **latitude**                    | Latitude of the listing                                                                        | Numeric            |
| **longitude**                   | Longitude of the listing                                                                       | Numeric            |
| **room_type**                   | Type of accommodation being offered                                                           | Categorical        |
| **price**                       | Price per night in USD                                                                         | Numeric            |
| **minimum_nights**              | Minimum number of nights required to book a listing                                            | Numeric            |
| **number_of_reviews**           | Total number of reviews a listing has                                                         | Numeric            |
| **last_review**                 | Date of the last review when the listing was rented                                            | Date               |
| **reviews_per_month**           | Total number of reviews a listing receives in a month                                          | Numeric            |
| **calculated_host_listings_count** | Number of listings per host                                                                 | Numeric            |
| **availability_365**            | Number of days a listing is available in a year                                               | Numeric            |
| **number_of_reviews_ltm**       | Total number of reviews a listing has in the last 12 months                                    | Numeric            |
| **city**                        | City where the listing is located                                                             | Categorical        |

## Data Preprocessing and EDA
The raw dataset contained missing values, irrelevant attributes, and outliers. The following steps were taken to clean and preprocess the data:

**Handling Missing Values:** Dropped columns with a high percentage of missing values, such as last_review, reviews_per_month, and neighbourhood_groups.

**Removing Irrelevant Columns:** Attributes like id, name, host_id, and host_name were excluded as they were not useful for prediction.

**Outlier Removal:** Price outliers were identified and removed using the interquartile range (IQR). For instance, listings with prices as low as $0 or as high as $100,000 were identified as errors.

**Feature Selection:** Retained columns like price, minimum_nights, number_of_reviews, availability_365, and room_type.

## Exploratory Data Analysis (EDA)
The cleaned dataset was analyzed to derive insights into pricing trends and other features. Below are the key findings:

### 1. Average Price Across Each City:
Geographic location significantly influences pricing. Cities like Santa Cruz County and Rhode Island have the highest average prices, while Columbus and Salem have the lowest.

![Average price across each city](Images/Average%20price%20across%20each%20city.png)


### 2. Number of Listings in Each City:
Listing density varies widely and impacts pricing competition. New York and Los Angeles have the highest number of Airbnb listings, while smaller cities have fewer listings.

![Number of Listings in Each City](https://github.com/sindhu28ss/airbnb-price-prediction-service/blob/main/Images/Number%20of%20Listings%20in%20each%20city.png)


### 3. Distribution of Listing Availability Across Cities:
Availability impacts demand and consequently pricing. Listings in NY and Austin are frequently reserved, while those in Rhode Island and Pacific Grove have lower availability.

![Distribution of Listings Availability Across Cities](https://github.com/sindhu28ss/airbnb-price-prediction-service/blob/main/Images/Distribution%20of%20listings%20availability%20across%20cities.png)


### 4. Relationship Between Price and Number of Reviews:
Higher prices often lead to lower engagement. Listings priced below $200 receive more reviews, while higher-priced listings receive fewer reviews.

![Price vs Reviews](https://github.com/sindhu28ss/airbnb-price-prediction-service/blob/main/Images/Prics%20vs%20Reviews.png)


### 5. Room Type Distribution Across Cities:
Room type plays a key role in pricing variations. Entire homes dominate the market, followed by private rooms. Hotel rooms have the highest median prices.

![Room type distribution across cities](https://github.com/sindhu28ss/airbnb-price-prediction-service/blob/main/Images/Romm%20type%20distribution%20across%20cities.png)


### 6. Locations of Cities on the Map:
Geographic clustering indicates areas of high demand. Listings are concentrated in metropolitan areas and tourist-heavy locations.

![Locations of the cities on the map](https://github.com/sindhu28ss/airbnb-price-prediction-service/blob/main/Images/map.png)


### Correlation Matrix
The correlation heatmap below illustrates the relationships between price and other numerical variables in the dataset. The heatmap highlights that:
- `number_of_reviews_ltm` and `number_of_reviews` have a moderate positive correlation.
- `price` has weak correlations with all the numerical variables.

![Correlation Heatmap: Price vs Numerical Variables](https://github.com/sindhu28ss/airbnb-price-prediction-service/blob/main/Images/Correlation%20Heat%20Map.png)

## Model Training:
In this phase, I focused on building predictive models by utilizing relevant features from the dataset, performing feature engineering, evaluating variable importance, and optimizing the model to ensure effective and interpretable predictions for Airbnb listing prices.

Categorical variables such as room_type and city were converted into dummy variables to enable their inclusion in the regression model.
This increased the predictor count to 34 variables due to the 4 room types and 27 cities. Partitioned the data into training and validation sets and trained a Multiple Linear Regression model.

### Base Model:

**Predictors in the Base Model:** minimum_nights, number_of_reviews, calculated_host_listings_count, availability_365, number_of_reviews_ltm, and dummy variables for room_type and city.

**Outcome Variable:** price.

**Performance:** Root Mean Squared Error (RMSE): 137.07.

## Variable Importance Analysis:

A Variable Importance plot was generated to evaluate the contribution of different variables, especially city variables, to the prediction task.
Less influential city variables were identified and excluded.

## Exhaustive Search and Backward Elimination:

Applied Exhaustive Search and Backward Elimination techniques to streamline the predictors and retain the most significant variables.
This resulted in a Final Model with only 5 predictor variables.

## Final Model:

**Predictors:** availability_365, calculated_host_listings_count, minimum_nights, number_of_reviews, room_type_Private room.

**Performance:** RMSE: 140.14.

The trade-off between RMSE and model simplicity was considered, prioritizing a parsimonious model for real-world applicability.

## Performance Comparison:

The final model reduced the number of predictors from 34 to 5, resulting in a slight increase in RMSE from 137.07 to 140.14.
Approximately 95% of residuals lie within the range of [-300, 300], showcasing reasonable accuracy.

## Residual Analysis:

Histogram plots were generated to visualize residual distribution, confirming minimal bias in predictions.




