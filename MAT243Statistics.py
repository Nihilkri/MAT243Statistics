import math
#print("Math loaded")
import numpy as np
#print("Numpy loaded")
import scipy.stats as st
#print("Scipy loaded")
import pandas as pd
#print("Pandas loaded")
import matplotlib.pyplot as plt
#print("Pyplot loaded")
import sklearn as sks
#print("Scikit-learn loaded")
import seaborn as sns
#print("Seaborn loaded")
import quandl as qn

def plotSine():
  """ Plots a basic sine wave """
  theta = chr(0x03B8)
  x = [th/10000.0 for th in range(math.ceil(2*np.pi*10000))]
  y = [np.sin(th) for th in x]
  
  plt.title("Sine function", fontsize=16)
  plt.xlabel(theta)
  plt.ylabel(f"sin({theta})")
  plt.plot(x, y)
  plt.show()


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


def abandc(abc:list) -> str:
  n:int = len(abc)
  if(n == 0): return s
  s:str = str(abc[0])
  if(n == 1): return s
  for a in range(1, n - 1):
    s += ", " + str(abc[a])
  s += ("," if n > 2 else "") + " and " + str(abc[-1])
  return s

def stats(dat:list):
  print("Data:", dat)

  n = len(dat)
  if(n == 0):
    print("No statistics on an empty set.")
    return

  sdat = sorted(dat)
  if(sum([(0 if dat[i] == sdat[i] else 1) for i in range(n)]) == 0):
    print("Data is sorted.")
  else:
    print("Sorted data:", sdat)

  print("Samples:", n)

  mean = sum(dat)/n
  print("Mean:", mean)

  # 1 2 3 4 5 6 7 8 9 A B C  |
  # L 0 a 0 M 0 b 0 H        n09 %2=1 %4=1 0q1 1q3.0 2q5.0 3q7.0 4q09
  # L 0 m 0 A B 0 m 0 H      n10 %2=0 %4=2 0q1 1q3.0 2q5.5 3q8.0 4q10
  # L 0 a b 0 M 0 a b 0 H    n11 %2=1 %4=3 0q1 1q3.5 2q6.0 q18.5 4q11
  # L 0 a b 0 A B 0 a b 0 H  n12 %2=0 %4=0 0q1 1q3.5 2q6.5 3q9.5 4q12
  q0, q4 = sdat[0], sdat[-1]
  if(n % 2 == 1):
    m = (n + 1) // 2
    q2 = sdat[m - 1]
    if(n % 4 == 1):
      a = (m + 1) // 2
      b = m + a - 1
      q1, q3 = sdat[a - 1], sdat[b - 1]
    else: # n % 4 == 3
      a, b = m // 2, m // 2 + 1
      q1 = (sdat[a - 1] + sdat[b - 1]) / 2
      a, b = a + m - 1, b + m - 1
      q3 = (sdat[a - 1] + sdat[b - 1]) / 2
  else: # n % 2 == 0
    a, b = n // 2, n // 2 + 1
    q2 = (sdat[a - 1] + sdat[b - 1]) / 2
    if(n % 4 == 2):
      m = (a + 1) // 2
      q1 = sdat[m]
      m += b - 1
      q3 = sdat[m]
    else: # n % 4 == 0
      c, d = a // 2, a // 2 + 1
      q1 = (sdat[c - 1] + sdat[d - 1]) / 2
      c, d = c + b - 1, d + b - 1
      q3 = (sdat[c - 1] + sdat[d - 1]) / 2
  print("Median:", q2)
  print("Q0", q0, ", Q1", q1, ", Q2", q2, ", Q3", q3, ", Q4", q4)

  print("Range:", q4 - q0)
  print("Skew:", mean - q2)
  iqr = q3 - q1
  print("Interquartile Range:", iqr)
  ol1 = q1 - 1.5 * iqr
  ol1l = [x for x in sdat if x < ol1]
  if(len(ol1l) == 0):
    print("No outliers <", ol1)
  else:
    print(f"Outliers <{ol1}: {ol1l}")
  ol2 = q3 + 1.5 * iqr
  ol2l = [x for x in sdat if x > ol2]
  if(len(ol2l) == 0):
    print("No outliers >", ol2)
  else:
    print(f"Outliers >{ol2}: {ol2l}")

  unique = {}
  for x in dat:
    if(x in unique):
      unique[x] += 1
    else:
      unique[x] = 1
  itm = sorted(list(unique.items()), key = lambda x: x[1])
  print("itm:", itm)
  modecount = itm[-1][1]
  if(modecount == 1):
    print("No modes, all unique values")
  else:
    modes = sorted([i[0] for i in itm if i[1] == modecount])
    print(f"Mode{("s" if len(modes) > 1 else "")}: {abandc(modes)}, with count {modecount}")

  sumsqd = sum([(x - mean) ** 2 for x in dat])
  print("Sum of the square differences:", sumsqd)
  popvar = sumsqd / n
  print("Population variance:", popvar)
  popstd = math.sqrt(popvar)
  print("Population standard deviation:", popstd)
  smpvar = sumsqd / (n - 1)
  print("Sample variance:", smpvar)
  smpstd = math.sqrt(smpvar)
  print("Sample standard deviation", smpstd)  
  mad = sum([abs(x - mean) for x in dat]) / n
  print("Mean absolute deviation", mad)


def section22(x:list, p:list):
  """ 2.2 Properties of discrete probability distributions """
  # x defines a list containing the outcomes in the sample space
  # p defines a list containing the probabilities for each outcome
    
  # Links the values in x to the probabilities in p
  discvar = st.rv_discrete(values=(x,p))

  mean = sum([x[i] * p[i] for i in range(len(x))])
  print("Mean:", mean, ",", discvar.mean())

  var = sum([(x[i] - mean) ** 2 * p[i] for i in range(len(x))])
  print("Variance:", var, ",", discvar.var())

  std = math.sqrt(var)
  print("Standard Deviation:", std, ",", discvar.std())

  return 0

  


if(__name__ == "__main__"):
  print("Imports loaded!")
  #stats([2, 3, 1, 1, 2, 4, 2, 1, 1, 3])
  #stats([3, 37, 23, 61, 36, 65, 6, 24, 1, 19, 72, 1, 13, 40, 1])
  #stats([1, 2, 8, 9])
  #x, p = [0,1,2,3,4,5,6], [0.1,0.2,0.3,0.1,0.1,0.0,0.2]

  x, p = [0, 100, 150], [0.2, 0.7, 0.1]
  section22(x, p)
  print("Goodbye!")
  