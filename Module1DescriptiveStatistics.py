from Basic import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def pa131():
  """ Python-Function 1.3.1: Unemployment rates.
  The code below loads a dataset containing unemployment rates in the United States from 1980 to 2017 and plots the data as a line chart. """
  # loads the unemployment dataset
  unemployment = pd.read_csv('http://data-analytics.zybooks.com/unemployment.csv')
  # title
  plt.title('U.S. unemployment rate', fontsize = 20)
  # x and y axis labels
  plt.xlabel('Year')
  plt.ylabel('% of total labor force')
  # plot
  plt.plot(unemployment["Year"], unemployment["Value"])
  # saves the image
  #plt.savefig("unemployment.png")
  # shows the image
  plt.show()


def pa132():
  """ Python-Function 1.3.2: Automobile collisions.
  The code below plots the number of automobile collisions related to speeding and the total number of collisions on the same set of axes.
  The data is presented as a stacked bar chart in which two bar charts are overlaid on each other. """
  # intialize figure
  f, ax = plt.subplots()
  # load dataframe
  crashes = sns.load_dataset("car_crashes")
  df = crashes.loc[range(5)]
  # plot total crashes
  sns.set_color_codes("pastel")
  sns.barplot(x="total", y="abbrev", data=df,
              label="Total", color="b")
  # plot crashes related to speeding
  sns.set_color_codes("muted")
  sns.barplot(x="speeding", y="abbrev", data=df,
              label="Speeding-related", color="b")
  # title
  plt.title('Speeding-related automobile collisions', fontsize=20)
  # legend
  ax.legend(ncol=1, loc="lower right")
  ax.set(xlim=(0, 28), ylabel="State", xlabel="Automobile collisions (per billion miles)");
  # saves the image
  #plt.savefig("stacked.png")
  # shows the image
  plt.show()


def pa141():
  # loads the titanic dataset 
  titanic = sns.load_dataset("titanic")
  #print("Rows: ", titanic.shape[0])
  #print("Columns Names: ", titanic.columns)
  #print("Column Dtypes: ", titanic.dtypes)
  print(titanic['age'].min())

def statstest():
  #stats([2, 3, 1, 1, 2, 4, 2, 1, 1, 3])
  #stats([3, 37, 23, 61, 36, 65, 6, 24, 1, 19, 72, 1, 13, 40, 1])
  #stats([1, 2, 8, 9])
  pass
