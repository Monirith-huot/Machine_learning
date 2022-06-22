# -*- coding: utf-8 -*-
"""Homework.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nts5XbsjNcHItUmqVNj9mYpmMXSvddma
"""

import sklearn #This is use for numpy
import pandas as pd  #Data management
import numpy as np  # computations

# 1. Load the dataset
dataset = pd.read_csv("https://raw.githubusercontent.com/datasets/browser-stats/master/data.csv")


# 2. Dimension of dataset

dataset.shape

dataset.describe()

# 3. Check the variable (Numberical / Categorial)

numberical_variable = dataset.columns[dataset.dtypes != 'object']
dataset[numberical_variable]

categorical_variable = dataset.columns[dataset.dtypes == 'object']
dataset[categorical_variable]


# 4 Missing value 

dataset.isnull()
dataset[numberical_variable].isnull().head(5)
dataset[categorical_variable].isnull().head(5)

# 5. Find out the percentanges of missing values 
dataset.isnull().sum()/ len(dataset)

# 6. Replace value 
dataset.fillna(1)

# 7 . Fillup the mission values 
dataset['Moz-All'].replace(7.3,1) 

# 8 Create the rows and add it with the original dataset

row_insert = pd.Series(['2022-05', 3.5, 72.6, 17.5, 5.7, 7.9, 5.2, 7.9, 1.2, 3,6], index = ["Date", "AOL", "Chrome", "Firefox", "Internet Explorer", "Moz-All", "Mozilla", "Netscape", "Opera", "Safari"])
dataset.append(row_insert, ignore_index= True)

#9 Choose exact row from the dataset
dataset.iloc[0] #ilog help us to get specific row based on the index

dataset.iloc[1:5]
dataset[1:17]

datasetValue = dataset.set_index(dataset['Chrome']) 
datasetValue.loc[70.4]

#10 Rename the column

dataset.rename(columns={'Chrome' : "Chromes"})

## 11. Find the min, max, sum, count
print("Max: Data " , dataset["Moz-All"].max())
print("Min:  Data", dataset["Moz-All"].min())
print("Count: Data", dataset["Moz-All"].count())
print("Total: data: ", dataset["Moz-All"].sum())


#12 Delete the row and column
dataset.drop("Internet Explorer", axis= 1)

dataset.drop(dataset.columns[1], axis= 1) #Delete the first column 
dataset.drop([1], axis = 0) #Delete row where index = 1

# 13 Unique value
dataset["Moz-All"].unique()
