# Airbnb-Price-Prediction
<p align="center">
  <img src="https://github.com/sindhu28ss/Rental-Offerings-Optimization/blob/main/Airbnb-logo.png" width="250">
</p>
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
<p align="left">
  <img src="https://github.com/sindhu28ss/airbnb-price-prediction-service/blob/main/Images/Average%20price%20across%20each%20city.png" width="500">
</p>

### 2. Number of Listings in Each City:
Listing density varies widely and impacts pricing competition. New York and Los Angeles have the highest number of Airbnb listings, while smaller cities have fewer listings.

<p align="left">
  <img src="https://github.com/sindhu28ss/airbnb-price-prediction-service/blob/main/Images/Number%20of%20Listings%20in%20each%20city.png" width="500">
</p>

### 3. Distribution of Listing Availability Across Cities:
Availability impacts demand and consequently pricing. Listings in NY and Austin are frequently reserved, while those in Rhode Island and Pacific Grove have lower availability.

<p align="left">
  <img src="https://github.com/sindhu28ss/airbnb-price-prediction-service/blob/main/Images/Distribution%20of%20listings%20availability%20across%20cities.png" width="500">
</p>

### 4. Relationship Between Price and Number of Reviews:
Higher prices often lead to lower engagement. Listings priced below $200 receive more reviews, while higher-priced listings receive fewer reviews.

<p align="left">
  <img src="https://github.com/sindhu28ss/airbnb-price-prediction-service/blob/main/Images/Prics%20vs%20Reviews.png" width="500">
</p>

### 5. Room Type Distribution Across Cities:
Room type plays a key role in pricing variations. Entire homes dominate the market, followed by private rooms. Hotel rooms have the highest median prices.

<p align="left">
  <img src="https://github.com/sindhu28ss/airbnb-price-prediction-service/blob/main/Images/Romm%20type%20distribution%20across%20cities.png" width="500">
</p>

### Correlation Matrix
The correlation heatmap below illustrates the relationships between price and other numerical variables in the dataset. The heatmap highlights that:
- `number_of_reviews_ltm` and `number_of_reviews` have a moderate positive correlation.
- `price` has weak correlations with all the numerical variables.

<p align="left">
  <img src="https://github.com/sindhu28ss/airbnb-price-prediction-service/blob/main/Images/Correlation%20Heat%20Map.png" width="500">
</p>

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

<p align="left">
  <img src="https://github.com/sindhu28ss/airbnb-price-prediction-service/blob/main/Images/Variable%20importance%20plot.png" width="500">
</p>
## Exhaustive Search and Backward Elimination:

Applied Exhaustive Search and Backward Elimination techniques to streamline the predictors and retain the most significant variables.
This resulted in a Final Model with only 5 predictor variables.

## Final Model:

**Predictors:** availability_365, calculated_host_listings_count, minimum_nights, number_of_reviews, room_type_Private room.

**Performance:** RMSE: 140.14.

![Regression Statistics](https://github.com/sindhu28ss/airbnb-price-prediction-service/blob/main/Images/Regression%20statistics.png)

The trade-off between RMSE and model simplicity was considered, prioritizing a parsimonious model for real-world applicability.

## Performance Comparison:

The final model reduced the number of predictors from 34 to 5, resulting in a slight increase in RMSE from 137.07 to 140.14.
Approximately 95% of residuals lie within the range of [-300, 300], showcasing reasonable accuracy.

## Residual Analysis:

Histogram plots were generated to visualize residual distribution, confirming minimal bias in predictions.

<p align="left">
  <img src="https://github.com/sindhu28ss/airbnb-price-prediction-service/blob/main/Images/Residuals.png" width="500">
</p>

## **Python Scripts**

- **`price_prediction.py`**: Contains the logic for data preprocessing, model training, evaluation, and saving the trained model.  
- **`price_prediction_model.pkl`**: Serialized file of the trained model, ready for deployment.

## Model Deployment:

The model is deployed using Flask, serving predictions through a RESTful API.

## Environment Setup and Dependency Management

**Pipenv Setup:** Used pipenv for managing Python dependencies.

**Installed essential libraries:** flask, pandas, scikit-learn, numpy, and scipy.

Created Pipfile and Pipfile.lock to ensure reproducibility.

Installed flake8 as a development dependency for code linting and gunicorn for production readiness.

## Containerization with Docker

Created a Dockerfile to containerize the Flask application. Built a Docker image and ran a container locally.

**Key Files:**
* Dockerfile: Contains instructions for building the Docker image.
* requirements.txt: Lists Python dependencies for the app.

### Docker Commands:

**Build the Docker image:** `docker build -t price-prediction-app`

**Run the Docker container:** `docker run -p 9696:9696 price-prediction-app`

**Test the app with a curl command:** `curl "http://127.0.0.1:9696/predict_price?availability_365=100&calculated_host_listings_count=2&minimum_nights=3&number_of_reviews=10&room_type_Private%20room=1"`

## Deployment on Kubernetes
Deployed the containerized Flask app to a Kubernetes cluster using Minikube.
* Tagged and pushed the Docker image to Docker Hub:
* Installed Kubernetes tools (kubectl and minikube) and started Minikube.
* Created Kubernetes configuration files: `deployment.yaml` for creating pods, `service.yaml` for exposing the application.
* Deployment commands: `kubectl apply -f deployment.yaml` `kubectl apply -f service.yaml`
* Accessed the app using Minikube: `minikube service price-service --url`

### Testing the Deployment
* Browser Access: Navigate to the Minikube service URL and confirm the welcome page is displayed.

  ![Browser Screenshot](https://github.com/sindhu28ss/airbnb-price-prediction-service/blob/main/Images/Browser.png)

* API Testing: Test the /predict_price endpoint using curl

  ![Terminal Curl Response](https://github.com/sindhu28ss/airbnb-price-prediction-service/blob/main/Images/Terminal%20curl%20response.png)





