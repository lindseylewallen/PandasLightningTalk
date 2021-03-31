# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 11:48:31 2021

@author: lkl444
"""

#import modules, functions, and objects
import pandas 
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

#retrieve dataset
from urllib.request import urlretrieve
iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
urlretrieve(iris)

#use pandas to read data
dataset = pandas.read_csv(iris, sep=',')

#define attributes
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
dataset.columns = attributes


#Explore data
print(dataset.head()) #to check the first 5 rows of the data set
print(dataset.sepal_length.head())#just look at one column
print(dataset['sepal_length'].head())#same command as above
print(dataset.tail()) #check out last 10 row of the data set
print(dataset.sample(5)) #pops up 5 random rows from the data set 
print(dataset.shape) #returns output as mxn
print(dataset.ndim) #returns an integer representing the number of dimensions
print(dataset.dtypes) #check datatypes of columns

#Explore statistics
print(dataset.describe()) #to give a statistical summary about the dataset
print(dataset['sepal_length'].sum()) #Look at statistics of one column
print(dataset['sepal_length'].max()) #Look at statistics of one column


#Cleaning Data
#Print number of null elements in the dataset
print(dataset.isnull().sum()) 
#drops a row if any of the values are missing. Can set to '1'for columns or 'all' to drop only if all values are missing
dataset.dropna(axis=0,how = 'any')

#Filter Data
print(dataset[dataset['class']=='Iris-virginica'])
print(dataset[dataset['sepal_length']>6.3])

#Rename columns and show first 5 rows
print(dataset.rename(columns={"sepal_length":"sepal_length_cm"}).head())
#Delete sepal length column and show first 5 rows
print(dataset.drop(columns='sepal_length').head())


#Visualize data
# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()