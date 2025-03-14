# DATA DESCRIPTION

## 1. _3 Stocks & Bitcoin_ ([stock](stock))

The dataset contains historical stock and cryptocurrency price data from various sources. It consists of 1,520 observations and 5 columns. The columns are as follows:

- **Date**: The date of the recorded price (in MM/DD/YYYY format).
- **AMZN**: The stock price of Amazon.
- **DPZ**: The stock price of Domino's Pizza.
- **BTC**: The price of Bitcoin.
- **NFLX**: The stock price of Netflix.

The dataset spans multiple years and includes daily price changes for these financial assets.

This dataset is intended for a regression task, where the goal is to predict future values of a target financial asset based on historical price trends. Possible applications include:

- Predicting the stock price of **Amazon (AMZN)**.
- Predicting the stock price of **Netflix (NFLX)**.
- Predicting the stock price of **Domino’s Pizza (DPZ)**.
- Predicting the price of **Bitcoin (BTC)**.

Each of these columns can be used as a target variable (dependent variable) for regression models. Features may include previous prices of all assets to identify trends and correlations.

The dataset can be used to train machine learning models such as linear regression, decision trees, and deep learning models for time-series forecasting.

---

### 2. _Tesla Stock Price_ ([tesla](tesla))

The dataset contains historical stock price data with 1,692 observations and 7 columns. The columns are as follows:

- **Date**: The date of the recorded stock prices (in MM/DD/YYYY format).
- **Open**: The opening price of the stock.
- **High**: The highest price reached during the trading session.
- **Low**: The lowest price recorded during the trading session.
- **Close**: The closing price of the stock.
- **Volume**: The number of shares traded during the day.
- **Adj Close**: The adjusted closing price after considering splits and dividends.

The dataset spans multiple years and captures daily price movements of a stock.

This dataset is designed for a regression task, aiming to predict future stock prices based on historical data. Potential objectives include:

- Predicting the **Closing price** (`Close`) of the stock.
- Predicting the **Adjusted Closing price** (`Adj Close`).
- Predicting the **High** and **Low** prices for the next trading session.

The dataset can be utilized to develop machine learning models for time-series forecasting, such as linear regression, decision trees, recurrent neural networks (RNNs), or long short-term memory (LSTM) models.

Features for prediction may include previous prices, trading volume, and price trends over time.

---

# 3. _Daily Delhi Climate_ ([delhi](delhi))

The dataset contains historical weather data with two separate files:

- **Train dataset**: 1,462 observations
- **Test dataset**: 114 observations

Each dataset consists of 5 columns:

- **date**: The date of the recorded weather data (in YYYY-MM-DD format).
- **meantemp**: The average daily temperature (°C).
- **humidity**: The average daily humidity (%).
- **wind_speed**: The average daily wind speed (m/s).
- **meanpressure**: The average daily atmospheric pressure (hPa).

The dataset spans multiple years and captures daily weather conditions.

This dataset is designed for a regression task, aiming to predict weather conditions based on historical data. Possible objectives include:

- Predicting the **mean temperature** (`meantemp`).
- Predicting the **humidity** levels.
- Predicting **wind speed** variations.
- Predicting the **atmospheric pressure** (`meanpressure`).

Each of these columns can be used as a target variable for regression models, with the other variables serving as features. Machine learning models such as linear regression, decision trees, and time-series forecasting methods like LSTMs can be applied.

The train dataset is used for model training, while the test dataset is reserved for evaluation.

---

# 4. _Superstore Sales_ ([sales](sales))

The dataset contains sales transaction records with 9,800 observations and 18 columns. The columns are as follows:

- **Row ID**: Unique identifier for each row.
- **Order ID**: Unique identifier for each order.
- **Order Date**: The date when the order was placed (in MM/DD/YYYY format).
- **Ship Date**: The date when the order was shipped (in MM/DD/YYYY format).
- **Ship Mode**: The shipping method used (e.g., Standard Class, Second Class).
- **Customer ID**: Unique identifier for each customer.
- **Customer Name**: The name of the customer.
- **Segment**: The market segment the customer belongs to (e.g., Consumer, Corporate).
- **Country**: The country where the order was placed.
- **City**: The city where the order was placed.
- **State**: The state where the order was placed.
- **Postal Code**: The postal code of the shipping destination.
- **Region**: The geographical region of the order (e.g., West, South).
- **Product ID**: Unique identifier for each product.
- **Category**: The broad product category (e.g., Furniture, Office Supplies).
- **Sub-Category**: A more specific classification of the product.
- **Product Name**: The name of the product.
- **Sales**: The total sales amount for the transaction.

The dataset spans multiple years and provides insights into customer transactions, product categories, and sales performance.

This dataset is designed for a regression task, aiming to predict **the total sales for the next day** based on historical data. The key objectives include:

- Predicting **next-day total sales** using past sales trends.
- Understanding the influence of factors such as **seasonality, customer segment, shipping mode, and product category** on sales fluctuations.
- Forecasting sales trends to optimize inventory and marketing strategies.

To predict the total sales for the next day, we can engineer the following features:

- **Past Sales Trends**: Aggregated daily sales from previous days (e.g., last 7 days, last 30 days).
- **Sales Moving Averages**: Rolling averages over different time windows (weekly, monthly).
- **Day of the Week**: Sales trends may vary depending on the weekday.
- **Order Volume**: The number of transactions per day.
- **Customer Segments**: The proportion of sales coming from different customer segments (Consumer, Corporate, Home Office).
- **Shipping Mode Influence**: The effect of different shipping methods on daily sales.
- **Seasonality Factors**: Month, quarter, and year information to capture seasonal trends.

---

## 5. _Covid-19 Cases_ ([covid](covid))

The dataset contains daily records of confirmed cases of an event (e.g., a disease outbreak) with 841 observations and 2 columns. The columns are as follows:

- **Date**: The date of recorded cases (in YYYY-MM-DD format).
- **Confirmed**: The number of confirmed cases on that date.

This dataset captures the time-series progression of confirmed cases over a specific period.

This dataset is intended for a regression task, aiming to predict **the number of confirmed cases for the next day** based on historical trends. Possible objectives include:

- Forecasting the **next day's confirmed cases** using past case trends.
- Identifying patterns and trends in the time-series data.
- Modeling the spread of cases for future projections.

To predict the next day's confirmed cases, potential features include:

- **Previous Daily Cases**: Using past confirmed cases (e.g., last 7 days, last 14 days) to detect trends.
- **Moving Averages**: Calculating rolling averages over different time windows (weekly, bi-weekly).
- **Day of the Week Effects**: Some events may show fluctuations depending on the weekday.
- **Growth Rate**: The rate of increase or decrease compared to previous days.

---

## 6. _Rain in Australia_ ([rainy](rainy))

The dataset contains weather observations from multiple locations over several years. It includes 145,460 observations and 23 columns, which are as follows:

- **Date**: The date of the recorded weather data (in YYYY-MM-DD format).
- **Location**: The name of the weather station location.
- **MinTemp**: The minimum temperature of the day (°C).
- **MaxTemp**: The maximum temperature of the day (°C).
- **Rainfall**: The amount of rainfall recorded (mm).
- **Evaporation**: The amount of evaporation (mm).
- **Sunshine**: The number of hours of sunshine during the day.
- **WindGustDir**: The direction of the strongest wind gust.
- **WindGustSpeed**: The speed of the strongest wind gust (km/h).
- **WindDir9am**: The wind direction at 9 AM.
- **WindDir3pm**: The wind direction at 3 PM.
- **WindSpeed9am**: The wind speed at 9 AM (km/h).
- **WindSpeed3pm**: The wind speed at 3 PM (km/h).
- **Humidity9am**: The humidity percentage at 9 AM.
- **Humidity3pm**: The humidity percentage at 3 PM.
- **Pressure9am**: The atmospheric pressure at 9 AM (hPa).
- **Pressure3pm**: The atmospheric pressure at 3 PM (hPa).
- **Cloud9am**: The cloud cover at 9 AM (measured in eighths).
- **Cloud3pm**: The cloud cover at 3 PM (measured in eighths).
- **Temp9am**: The temperature at 9 AM (°C).
- **Temp3pm**: The temperature at 3 PM (°C).
- **RainToday**: Whether it rained today (`Yes` or `No`).
- **RainTomorrow**: Whether it will rain tomorrow (`Yes` or `No`) (Target for classification).

The dataset includes numerical and categorical features related to daily weather conditions.

This dataset can be used for both **regression** and **classification** tasks.

### a. Classification Task: Predicting Rain Tomorrow

The objective is to classify whether it will rain tomorrow (`Yes` or `No`) based on weather conditions observed today. This is a **binary classification** problem where the target variable is:

- **RainTomorrow** (`Yes` or `No`)

Possible approaches include:

- Decision Trees, Random Forest, XGBoost, or Logistic Regression for binary classification.
- Deep learning methods like artificial neural networks (ANNs) for more complex patterns.

### b. Regression Task: Predicting Weather Metrics

The objective is to predict continuous weather variables based on past weather conditions. Possible target variables for regression include:

- **MaxTemp**: Predicting the maximum temperature for the next day.
- **MinTemp**: Predicting the minimum temperature for the next day.
- **Rainfall**: Predicting the amount of rainfall in mm.
- **WindGustSpeed**: Predicting the strongest wind gust speed.

Possible regression models include:

- Linear Regression, Random Forest Regressor, XGBoost for weather prediction.
- Time-series models such as ARIMA or deep learning models like LSTMs for temporal dependencies.
