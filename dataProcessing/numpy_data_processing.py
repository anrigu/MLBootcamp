# -*- coding: utf-8 -*-
"""Numpy/Data Processing

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aaznBQvQMmOsxS0hFBEeGbvmWeR9whZf

# Numpy and Data Processing using Pandas
### Numpy - a framework that imports lots of mathematical features (special functions etc.)
### Pandas - a framework for data processing. It has a lot of features that allow for importing and reading of csv files and plotting that data

I'll be going over these two libraries in this module. I wouldn't worry too much about this module EXCEPT for the one section on data uploading (which I labelled below). We probably won't be using it that much for this project but it's still a very useful library for ML applications, so I'll just add a quick intro for it here.

## Numpy
"""

# create a numpy array from this list
import numpy as np
''' How is this different from a normal python list? Numpy arrays are generally faster when it comes to manipulating them. 
If you want to represent a matrix (If you haven't taken linear algebra, then don't worry about this!) and then want to transpose it or find the dot product you can do so through built-in
functions available through numpy. 
'''
a = [1,2,3,4,5]
b = np.array(a)

# find the mean of b
mean_b = np.mean(b)
print(mean_b)

# get a list where each entry in b is squared (so the new numpy array is [1, 4, 9, 16, 25, 36])
# use a different (numpy-specific) approach
d = np.square(b)
print(d)

# change b from a length-6 list to a 2x3 matrix
b.resize((2,3))
print(b)

"""## Pandas

The section below is the only **crucial** one!
"""

# load in the "starbucks.csv" dataset
# This import statement imports the pandas library and creates an alias "pd" so that whenever you want to use it, you can say pd.[command] instead of pandas.[command]
import pandas as pd
import os
data = pd.read_csv('./sample_data/starbucks.csv')
# Prints a preview of the data (i.e. the first 5 data entries)
print(data.head())

# what is the average # calories across all items?
avg_cals = data["calories"].mean()
print("average calories across all items is: ", avg_cals)

# how many different categories of beverages are there?
diff_cats = data["beverage_category"].unique()
print("diff categories of beverages: ", len(diff_cats))

# plot the distribution of the number of calories in drinks using matplotlib
import matplotlib as plt
data.plot.scatter( x='beverage', y='calories')

# plot the distribution of calories in Short, Tall, Grande, and Venti drinks
# (you can use multiple lines for this)
# twist: you should also include the Nonfat Milk drinks that also have an associated size.

# you can decide how you want to visualize this. Colors? Small multiples? Density vs histogram?
# the starbucks is your oyster.
fets = ['Short', 'Tall', 'Grande', 'Venti', 'Short Nonfat Milk', 'Tall Nonfat Milk', 'Grande Nonfat Milk', 'Venti Nonfat Milk']
for fet in fets:
    data.loc[data['beverage_prep'] == fet]["calories"].plot.hist()