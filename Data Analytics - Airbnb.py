#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly as ply
import re
import sklearn as skl


# In[2]:


ds=pd.read_csv("C:/Users/duapr/OneDrive/Documents/Conestoga College Study/Predictive Analytics/Assignments records/Programming in Python/Python Project data set and coding/airbnb_listings_usa-edited.csv",low_memory=False)


# In[3]:


ds.head(10)


# In[4]:


ds.shape


# In[5]:


ds.ndim


# In[6]:


ds.axes


# Considering the column names of this dataset, we wish to describe what each column names mean:
# 
# <b>name</b> : Name of the listing on airbnb as posted by the host (<i>person putting their property on rent</i>)
#     Key features: interesting facts about the listing like listing site description, activities, etc.
# 
# <b>host_id</b> : Unique identifier for the airbnb host who has listed his/her property
# 
# <b>host_name</b> : name of the host
# 
# <t><b>neighbourhood_group</b> : The neighbourhood group as geocoded using the latitude and longitude
# 
# <b>neighbourhood</b> : actual location of the listing
# 
# <b>latitude</b> : 
# 
# <b>longitude</b> :
# 
# <b>room_type</b> : 'Entire home/apt', 'Private room', 'Shared room', 'Hotel'
# 
# <b>price</b> : Daily price of the listing in USD
# 
# <b>minimum_nights</b> : Minimum number of night stays of the listing the customer can book
# 
# <b>number_of_reviews</b> : total reviews the listing has received till last review date
# 
# <b>last_review</b> : date when the listing was last reviewed
# 
# <b>reviews_per_month</b> :
# 
# <b>calculated_host_listings_count</b> : The total number of listings the host has in the city/region area
# 
# <b>availability_365</b> :The availability of the listing 365 days in advance (determined by the calendar); meaning the listing
#                          might have been booked or blocked by the host.
# 
# <b>number_of_reviews_ltm</b> : Total number of reviews in last 12 months
#     
# <b>license</b>
# 
# <b>State</b> US State abbreviated for the state the listing is in 
# 
# <b>City</b> City the neighbourhood belongs to
#     
# 
#     
# Some wranling and cleaning will be needed as we see in the last output we got below abnormalities :
#     
#  1) Unamed column must be deleted 
# 2) Some abnormalities in the host name with foreign characters like special characters, non-ASCII characters, etc needs to be cleaned

# In[7]:


#showing datatypes
ds.dtypes


# # DATA CLEANING

# In[8]:


def Null_values():
    Nanvalues={}
    for i in ds.columns:
        null_count=ds[i].isna().sum()
        if null_count!=0:
            print("number of NaN values in {} is : {}".format(i,ds[i].isna().sum()))
Null_values()


# In above code,we have pulled up the the NaN valuescount  in the dataset only in attributes(i.e., column names) that we need for data analysis and modeling. 

# Below Code shows the functions we have used to cleanup the column "host_name"
# 
# Host name had special characters , numeric values and non ascii characters that should not be there. So we created functions to replace as done below :

# In[9]:


#Function to identify special characters using regex
def find_special_characters(text):
    special_characters = r'[-+&!@#$%^&*/(),.?":{}|<>]'
    matches = re.findall(special_characters, str(text))  # Convert to string
    return matches

#Function to identify numeric characters using regex
def has_numeric_value(input_string):
    return bool(re.search(r'\d', str(input_string)))  # Convert to string

#Function to identify non ASCII characters using regex
def find_NONASCII_characters(text):
    special_characters = r'[^\x00-\x7F]'  # Matching non-ASCII characters
    matches = re.findall(special_characters, str(text))
    return matches

#Loop to fetch numeric characters using regex
for index, row in ds.iterrows():
    if len(find_special_characters(row['host_name'])) > 0:
            ds.at[index, 'host_name'] = 'HSTNM_' + str(row['host_id']) + '_' + str(row['calculated_host_listings_count'])
    elif has_numeric_value(row['host_name']):
        ds.at[index, 'host_name'] = 'HSTNM_' + str(row['host_id']) + '_' + str(row['calculated_host_listings_count'])
    elif len(find_NONASCII_characters(row['host_name'])) > 0:
        ds.at[index, 'host_name'] = 'HSTNM_' + str(row['host_id']) + '_' + str(row['calculated_host_listings_count'])


# In[10]:


#filling NaN values in reviews_per_month as 0 since we do not have any reviews for those listings.
ds['reviews_per_month'].fillna(0,inplace=True)

#filling NaN values in neighbourhood_group as Other 
ds['neighbourhood_group'].fillna('Other',inplace=True)


# # DATA WRANGLING

# In[11]:


#Dropping off the rows which contain blank cells in the "neighbourhood" column
ds = ds.dropna(subset=['neighbourhood'])

# Dropping off the rows with numeric integer values in the "neighbourhood" column
ds = ds[~ds['neighbourhood'].astype(str).str.isdigit()]


# In[12]:


ds=ds.drop(axis=1,columns=["Unnamed: 0", "license"])


# In[13]:


### Removing duplicate rows seen in the listings 
ds=ds.drop_duplicates()
ds.shape


# In[14]:


ds.head(10)


# In[15]:


#Extracting Los Angeles data from the dataset to zoom into Los Angeles neighbourhoods so that data can be aggregated easily and new features can be uncovered in the city 
Reviews_LA=ds.loc[(ds['city']=='Los Angeles')]
Reviews_LA


# In[27]:


Reviews_LA=Reviews_LA.loc[(Reviews_LA['price']<=1000) & (Reviews_LA['neighbourhood_group']!='')]
Reviews_LA


# In[26]:


#Average reviews per listing on the dataset for Los angeles
Reviews_LA['Average_reviews']=Reviews_LA['number_of_reviews']/Reviews_LA['calculated_host_listings_count']
Reviews_LA.head()


# # INSIGHT
# 
# <b>Note</b>: The Average_reviews column is used to average out the duplicate number of reviews that occur in listings where the calculated_host_listings_count is more than 1. 
# 
# Example : For a listing with calculated_host_listings_count=2 has number_of_reviews as 10 twice because there are two listings of the host , hence to avoid counting 10 as number of reviews as 20 , we instead divide both the rows by calculated_host_listings_count=2 and hence we will consider total reviews as 10 only.

# # DESCRIPTIVE ANALYTICS

# In[28]:


ds.describe()


# Using the describe command we are showing the summary of statistical data for all the numeric type attributes as shown below :
# 
# <b>Mean</b> - Average number that denotes the average usage or average avaialbiltiy across different attributes.
# <b>std</b> - it is the sum of the deviations of all observed values with the mean, it shows how the datapoints of a given attributes tesis varying around mean point of the same attribute
# 
# <b>min</b> - shows the lowest value of the observation in a given attribute <br>
# <b>max</b> - shows the highest value of the observation in a given attribute <br>
# <b>25%</b> - this is the 1st quantile which shows trend of how first 25% observations are following in the same attribute or column name<br>
# <b>50%</b> - this is the 1st quantile which shows trend of how first 50% observations are following in the same attribute or column name<br>
# <b>75%</b> - this is the 1st quantile which shows trend of how first 75% observations are following in the same attribute or column name<br><br>
# 
# Some examples of useful information from  instance :
# 
# 1. $280.17 is average of minimum number of nights that are available for booking across all neightbourhoods.<br>
# 2. 25% of all the listing have reviews less than equal to 1.<br>
# 3. 25% of the rooms listed have availability between 57 night to 175 nights in the next 1 year. <br>

# In[19]:


plt.figure(figsize=(10, 10))
sns.countplot(data=ds, x='room_type', hue='city', palette='plasma')


# In[20]:


ds.neighbourhood_group = ds.neighbourhood_group.astype('category')
ds.neighbourhood_group.cat.categories

pd.crosstab(ds.neighbourhood_group, ds.room_type)


# # Insight
# <br><br>
# <heading>
# In Los Angeles (City of Los Angeles), the distribution of Airbnb listings across different room types is as follows:
# 
# Entire home/apartment: There are 29,014 listings available, making it the most common type of accommodation option in the city.
# <br>Hotel room: There are 54 hotel rooms listed for rent.
# <br>Private room: There are 8,800 listings available for renting private rooms.
# <br>Shared room: There are 1,134 shared rooms listed for rent.
# <br><br>This summary provides an overview of the number of listings available for each room type in Los Angeles. It indicates that entire home/apartments are the most prevalent option, followed by private rooms, shared rooms, and hotel rooms, which are the least common.

# In[21]:


#catplot room type and price

filtered_ds = ds[ds['neighbourhood_group'] == 'City of Los Angeles']
plt.figure(figsize=(10,10))
sns.catplot(x="room_type", y="price", palette="rocket", data=filtered_ds);


# In[22]:


# create countplot roomtype and neighbourhood type (los Angeles)

plt.figure(figsize=(10,5))
countplot = sns.countplot(data=filtered_ds, x='room_type', palette='plasma', label='Room Types')


# In[23]:


#boxplot neighbourhood_group(Los Angeles) and room availability
plt.figure(figsize=(5,5))
boxplot = sns.boxplot(data=filtered_ds,y='availability_365',palette='cubehelix')
plt.xlabel('City of Los Angeles')
plt.ylabel('Availability (in days)')
plt.tight_layout()
plt.show()


# In[24]:


Reviews_LA.describe()


# In[29]:


Reviews_LA['price'].groupby(Reviews_LA['room_type']).mean()
sns.barplot(Reviews_LA,x='city',y=Reviews_LA['price'], hue='room_type')
#show numbers with the label or in the bars as well.


# In[30]:


Reviews_LA['price'].groupby(Reviews_LA['room_type']).mean()


# In[31]:


#Dropping  number of_reviews column as we use average_reviews column instead, hence dropping 
Reviews_LA=Reviews_LA.drop(axis=1,columns=['number_of_reviews'])
Reviews_LA.head()


# In[32]:


plot1=Reviews_LA['Average_reviews'].groupby(Reviews_LA['room_type']).count().reset_index()
sum=plot1['Average_reviews'].sum()
plot1['Percentage_reviews']=(plot1['Average_reviews']/sum)*100
sns.barplot(data=plot1,x='room_type',y='Percentage_reviews')


# # INSIGHT
# Based on plot of %reviews per room_type, it seems the popularity is for Entire home/apt is highest, followed by  Private room while Hotel room and Shared room are not that popular in the Los Angeles city

# In[33]:


#Average reviews by neighbourhood
Top20_NH_LA=Reviews_LA['Average_reviews'].groupby(Reviews_LA['neighbourhood']).sum().reset_index().sort_values(by='Average_reviews',ascending=False)
Top20_NH_LA.reset_index()
Top20_NH_LA=Top20_NH_LA.iloc[0:20]


# In[34]:


sns.barplot(data=Top20_NH_LA,x='Average_reviews',y='neighbourhood')


# # INSIGHT
# 
# <heading> From the number of reviews , we will consider the top 5 neighbourhoods = (Venice, Long Beach, Santa Monica, Silver Lake, Hollywood Hills) and perform further data analysis on how thes neighbourhoods ratings vary by room_type

# In[35]:


Top20_NH_LA


# In[36]:


#Extracting dataframe for top 5 neighbourhoods in Los Angeles uncovered above
Top5_NH_LA_Df=Reviews_LA.loc[(Reviews_LA['neighbourhood']=='Venice') | (Reviews_LA['neighbourhood']=='Long Beach') | (Reviews_LA['neighbourhood']=='Santa Monica') | (Reviews_LA['neighbourhood']=='Silver Lake') | (Reviews_LA['neighbourhood']=='Hollywood Hills')]
Top5_NH_LA_Df.head()


# # INSIGHT
# 
# <Heading> This is the same conclusion in countplot above which shows the avaialbility of Entire home or Private rooms to be more in demand as compared to other room types like Hotel and Shared room in top 5 neighbourhoods as well. Hence , it is logical to consider that this trend is same across all neighbourhoods.
# 
# <b> Hence, it should be OK to consider the listings with room type of Hotel Room and Shared Room less popular than the other two room types

# In[37]:


plt.figure(figsize=(10, 10))
sns.countplot(data=Top5_NH_LA_Df, x='room_type', hue='neighbourhood', palette='plasma')


# In[38]:


Top5_NH_LA_Df=Top5_NH_LA_Df.reset_index()


# In[57]:


Top5_NH_LA_Df


# In[59]:


Top5_NH_LA_Df.drop(axis=1,columns=['index'])


# In[62]:


sns.countplot(data=Top5_NH_LA_Df, x="neighbourhood",hue='room_type',palette="Spectral")


# In[70]:


#Tabulation of Average number of reviews for each neighbourhood
Top5_NH_LA_Df['Average_reviews'].groupby(Top5_NH_LA_Df['neighbourhood']).sum().reset_index()


# In[73]:


#Tabulation of mean number of reviews for each neighbourhood
NH_review2=Top5_NH_LA_Df['Average_reviews'].groupby(by=[Top5_NH_LA_Df['neighbourhood']]).sum().reset_index()
NH_review2.sort_values(by='neighbourhood')


# In[74]:


plt.pie(x=NH_review2['Average_reviews'],labels=NH_review2['neighbourhood'],autopct='%1.2f%%')


# In[75]:


ds['city'].unique()


# In[76]:


#Remove rows where price is shown as 0 as this can be bad data and will become outliers later on , so making sure that the  plot and predictions are not impacted by outliers


# # Price Predictions for Los Angeles
# 
# <heading> Below we are showing the price predictions for Los Angeles to fit an appropriate regression model since Regression for overall dataset didnt give much accuracy of predictions

# In[39]:


Correlation_matrix=ds.corr(numeric_only=True)
Correlation_matrix


# In[40]:


sns.heatmap(Correlation_matrix)


# As shown in above heatmap, the Correlation is very weak if observed on the entire dataset. To test accuracy of predictions, we will plot based on correlation shown between below attributes in the dataset :
# 
# <b>number of reviews<\b> : minimum nights, price, calculated_host_listings_count,availability_365<br>
# <b>price<\b> : latitude, longitude, reviews_per_month, minimum_nights, reviews_per_month, calculated_host_listings_count, availability_365
# 

# In[85]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# In[81]:


# Drop columns with a high percentage of missing values
Reviews_LA.drop(columns=['neighbourhood_group'], inplace=True)

# Impute missing values in 'reviews_per_month' with mean
Reviews_LA['reviews_per_month'].fillna(Reviews_LA['reviews_per_month'].mean(), inplace=True)

# Impute missing values in other columns as needed, or drop them if appropriate
# For example:
# ds['name'].fillna('Unknown', inplace=True)
# ds['host_name'].fillna('Unknown', inplace=True)
# ds['neighbourhood'].fillna('Unknown', inplace=True)
# ds['last_review'].fillna(ds['last_review'].mode()[0], inplace=True)

# Check if there are any remaining missing values
print(Reviews_LA.isnull().sum())

# Impute missing values in 'reviews_per_month' with mean
Reviews_LA['reviews_per_month'].fillna(Reviews_LA['reviews_per_month'].mean(), inplace=True)

# Impute missing values in other columns as needed, or drop them if appropriate
# For example:
# ds['name'].fillna('Unknown', inplace=True)
# ds['host_name'].fillna('Unknown', inplace=True)
# ds['neighbourhood'].fillna('Unknown', inplace=True)
# ds['last_review'].fillna(ds['last_review'].mode()[0], inplace=True)

# Check if there are any remaining missing values
print(Reviews_LA.isnull().sum())


# In[82]:


sns.pairplot(Reviews_LA, vars=['price','Average_reviews','minimum_nights','reviews_per_month','room_type'])


# In[83]:


Reviews_LA.columns


# In[84]:


Correlation_matrix_LA=Reviews_LA.corr(numeric_only=True)
Correlation_matrix_LA


# In[86]:


sns.heatmap(Correlation_matrix_LA)


# # LINEAR REGRESSION PRICE PREDICTION MODEL 

# In[87]:


#Selecting the features and target variable
yl=Reviews_LA['price']
xl=Reviews_LA[['latitude', 'longitude', 'Average_reviews','minimum_nights']]


# In[88]:


# Splitting the dataset into training and testing sets and evaluation of the model

x_train, x_test, y_train, y_test = train_test_split(xl, yl, test_size=0.2, random_state=42)

regr1 = LinearRegression().fit(x_train, y_train)
print("Coefficients:",regr1.coef_)
y_pred=regr1.predict(x_test)

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))


# In[89]:


#Plotting the Linear Model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Predicted')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Linear Regression Model: Actual vs. Predicted Prices')
plt.legend()
plt.show()


# # RANDOM FOREST REGRESSION MODEL

# In[90]:


# Selecting the features and target variable
X = Reviews_LA[['latitude', 'longitude', 'Average_reviews','minimum_nights']]
y = Reviews_LA['price']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[91]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# In[92]:


# Initializing and training the Random Forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the testing set
predictions = model.predict(X_test)


# In[93]:


# Evaluation of the model
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)


# In[94]:


# Plotting actual vs predicted prices of the Random Forest Regression Model
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions, alpha=0.5)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Random Forest Model with Actual vs Predicted Prices')
plt.grid(True)
plt.show()


# In[95]:


#Evaluation of the model 
from sklearn.metrics import mean_squared_error

# Calculate MSE
mse = mean_squared_error(y_test, predictions)

print("Mean Squared Error (MSE):",mse)


# In[96]:


# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(y_test, predictions))


# In[65]:


predictions


# In[97]:


X_test


# <b>PRICE PREDICTION MODEL USING THE CATEGORICAL VARIABLE <br><br>
# The code below represents the Price Prediction Model for Los Angeles, where the categorical variable "room type " is included along with other features like 'latitude, ''longitude,' 'minimum_nights', and 'Average_reviews'.

# In[98]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error

# Encoding the categorical variable 'room_type'
encoder = OneHotEncoder(sparse=False)
room_type_encoded = encoder.fit_transform(Reviews_LA[['room_type']])

# Selecting the features which have an impact on the target variable
X_numerical = Reviews_LA[['latitude', 'longitude', 'minimum_nights', 'Average_reviews']]
X = np.concatenate([X_numerical, room_type_encoded], axis=1)
y = Reviews_LA['price']

# Splitting of the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # Adjust test_size as needed

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Initializing and training the Random Forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Making the predictions on the testing set
predictions = model.predict(X_test)

# Calculation of the Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)


# In[99]:


# Calculating the R-squared value
r2 = r2_score(y_test, predictions)
print("R-squared value:", r2)


# In[100]:


# Plotting the model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price')
plt.show()


# <b> HISTOGRAM WITH KDE PLOT DEPICTING THE DISTRIBUTION OF PRICES BY ROOM TYPE

# In[101]:


import seaborn as sns
import matplotlib.pyplot as plt

# Colors being defined for each room type
colors = {'Entire home/apt': 'blue', 'Private room': 'green', 'Shared room': 'orange'}

# Plotting a histogram with KDE plot for each room type
plt.figure(figsize=(10, 6))
for room_type in Reviews_LA['room_type'].unique():
    if room_type in colors:
        sns.histplot(Reviews_LA[Reviews_LA['room_type'] == room_type]['price'], bins=20, kde=True, color=colors[room_type], label=room_type)

plt.title('Distribution of Prices by Room Type')
plt.xlabel('Price ($)')
plt.ylabel('Density')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('price_vs_roomtype_histogram_with_kde.png')  
plt.show()


# 

# In[ ]:




