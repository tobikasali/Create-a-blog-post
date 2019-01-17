# Create-a-new-blog-post
Udacity Term 2 Project

## Create a blog post

### Project Description

The purpose of this project is to demonstrate our ability to prepare data, work with a data set and follow the data science process.The goal is to choose a dataset, prepare the data, perform data analysis and communicate the results. 

### Contents of the Repository

Readme.md:  This file, describing the contents of this repo.

AirBnB.ipynb: The Jupyter notebook containing the analysis performed.

Sources files:
* s_listings.csv - The csv file containing the Seattle AirBnB listings.
* b_listings.csv - The csv file containing the Boston AirBnB listings.

Python Libraries:

* numpy and pandas - For data manipulations
* os and gmaps - To set system variables and enable use of gmaps.
* matplotlib.pyplot and seaborn - For showing plots
* sklearn linear_model, LinearRegression, model_selection, train_test_split, metrics,r2_score - To perform machine learning and check accuracy.

### Summary of Data sets and results

The AirBnB dataset contains listings of AirBnB properties in Seattle and Boston. I chose to try to answer 3 questions with the dataset.

1. What areas have the highest priced AirBnB homes? 

Based on the data,the highest earning homes per person are located along the water front in Seattle in the zipcode of 98104.

2. What do the different types of properties earn on Average? How do prices differ between between Seattle and Boston?

There doesn't seem to be a lot of differences in price for the different property types. However, For most of them, Boston is more expensive than Seattle. This difference is especially pronounced when it comes to Dorm rooms.

There was a similar trend in the prices per room as Boston is more expensive that Seattle for all property types that they both have. There are some property types that exist in once city but not the other. For example, Bungalows, Cabins or Chalets, tents or Treehouses only exist in Seattle while you have Villas and Guesthouses in Boston. Seattle certainly seems to have more diverse property types.

3. What are factors that mostly affect AirbnB prices?

The top 5 features contributing to price and they are shown below.

* bathrooms
* review_scores_location
* room_type_Private room
* room_type_Shared room
* cancellation_policy_strict

* The number of bathrooms - This makes sense because private bathrooms in apartment shares always make a difference in the price.

* review_scores_location - This is certianly interesting as well - as the best placed listings would probably have higher prices. 

* room_type_Private room
* room_type_Shared room

Again, I can see how these two features will play an important role in determining the price for a share.

* cancellation_policy_strict - Just like hotels, the stricter the cancellation policy the higher the price!

The steps followed can be found in jupyter notebook in this repository. 
