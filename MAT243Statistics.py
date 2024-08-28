import math
#print("Math loaded")
# import numpy as np
# #print("Numpy loaded")
import scipy.stats as st
# #print("Scipy loaded")
# import pandas as pd
# #print("Pandas loaded")
# import matplotlib.pyplot as plt
# #print("Pyplot loaded")
# import sklearn as sks
# #print("Scikit-learn loaded")
# import seaborn as sns
# #print("Seaborn loaded")
# import quandl as qn

sigma = chr(963)

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

def samplemeanstd(dat:list):
  n = len(dat)
  mean = sum(dat) / n
  sumsqd = sum([(x - mean) ** 2 for x in dat])
  smpstd = math.sqrt(sumsqd / (n - 1))
  return mean, smpstd

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

def statstest():
  #stats([2, 3, 1, 1, 2, 4, 2, 1, 1, 3])
  #stats([3, 37, 23, 61, 36, 65, 6, 24, 1, 19, 72, 1, 13, 40, 1])
  #stats([1, 2, 8, 9])
  pass


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

def section22test():
  #x, p = [0,1,2,3,4,5,6], [0.1,0.2,0.3,0.1,0.1,0.0,0.2]
  x, p = [0, 100, 150], [0.2, 0.7, 0.1]
  section22(x, p)

def generateallpermutations(dat:list, sampn:int, dbug:bool = False) -> list:
  popn = len(dat)
  perms = []
  samples = []
  a = [i for i in range(popn)]
  popperm = int(math.factorial(popn))
  print("Popperm =", popperm)
  sampperm = popperm // int(math.factorial(popn-sampn) )
  print("Sampperm =", sampperm)
  adat = [dat[a[i]] for i in range(popn)]
  if(dbug): print("adat =", adat)
  perms.append(adat)
  samples.append(adat[:sampn])
  for p in range(1, popperm + 1):
    if(dbug): print("Permutation", p, "/", popperm)
    l = k = popn - 2
    for k in range(popn - 2, -1, -1):
      if a[k] < a[k + 1]: break
    else:
      k = -1
    if(k < 0): break
    if(dbug): print("k =", k)
    for l in range(popn - 1, k, -1):
      if a[k] < a[l]: break
    if(dbug): print("l =", l)
    a[k], a[l] = a[l], a[k]
    for i in range(0, (popn - k - 1) // 2):
      a[k + 1 + i], a[popn - 1 - i] = a[popn - 1 - i], a[k + 1 + i]
    if(dbug): print("a =", a)
    adat = [dat[a[i]] for i in range(popn)]
    if(dbug): print("adat =", adat)
    #if(adat[sampn - 1] != perms[-1][sampn - 1]):
    if(sum([0 if adat[i] == perms[-1][i] else 1 for i in range(sampn)]) != 0):
      samples.append(adat[:sampn])
    perms.append(adat)
  if(dbug): print("Perms =", perms)
  if(dbug): print("Samples =", perms)
  return samples

def allsamples(dat:list, sampn:int, dbug:bool = False) -> list:
  popn = len(dat)
  samples = []
  a = [i for i in range(sampn)]
  numsamples = int(math.factorial(popn) // (math.factorial(popn-sampn) * math.factorial(sampn)))
  for i in range(numsamples):
    if(dbug): print("a =", a)
    adat = [dat [a[j]] for j in range(sampn)]
    if(dbug): print("adat =", adat)
    samples.append(adat[:])
    a[-1] += 1
    if(dbug): print("a =", a)
    for j in range(sampn - 1, 0, -1):
      if(a[j] == popn + j - sampn + 1):
        if(dbug): print(f"a[{j}] = {a[j]}")
        a[j - 1] += 1
        if(dbug): print("a =", a)
        for k in range(j, sampn):
          a[k] = a[j - 1] + k - j + 1
          if(dbug): print("a =", a)
    if(dbug): print()
  return samples


def samplingdistribution(dat:list, sampn:int):
  popn = len(dat)
  popmean, popstd = samplemeanstd(dat)
  SE = popstd / math.sqrt(popn)
  correctionfactor = math.sqrt((popn - sampn) / (popn - 1))
  samples = allsamples(dat, sampn)

  return samples

def zscore(x:float, mean:float, std:float) -> float:
  """ A z-score is a signed value that indicates the number of 
      standard deviations a quantity is from the mean """
  return (x - mean) / std

def section24(mean:float, std:float):
  """ Normal Distribution """
  for i in range(1, 6):
    lval = st.norm.cdf(i, mean, std)
    rval = st.norm.cdf(-i, mean, std)
    lrval = (lval - rval) * 100
    lstr = f"{-i}{sigma}={mean - i * std}"
    rstr = f"{i}{sigma}={mean + i * std}"
    print(f"{lrval:.6f}% of samples fall between {lstr} and {rstr}")

  print()
  for i in range(1, 6):
    lval = st.norm.cdf(i, mean, std)
    rval = st.norm.cdf(i - 1, mean, std)
    lrval = (lval - rval) * 100
    lstr = f"{i - 1}{sigma}={mean + (i - 1) * std}"
    rstr = f"{i}{sigma}={mean + i * std}"
    print(f"{lrval:2.6f}% of samples fall between {lstr} and {rstr}")
  print(f"{round((st.norm.cdf(100, mean, std) - st.norm.cdf(5, mean, std)) * 100, 6)}% of samples are greater than 5{sigma}={mean + 5 * std}")

  print()
  print(f"68% of samples are within {st.norm.ppf(0.5-0.34134475, mean, std)}{sigma} and {st.norm.ppf(0.5+0.34134475, mean, std)}{sigma}")
  print(f"50% of samples are within {st.norm.ppf(0.5-0.25, mean, std)}{sigma} and {st.norm.ppf(0.5+0.25, mean, std)}{sigma}")
  print(f"99% of samples are below {st.norm.ppf(0.99, mean, std)}{sigma}")
  print(f"99% of samples are above {st.norm.isf(0.99, mean, std)}{sigma}")


def section24test():
  pass
  #section24(511, 120)
  
  # dat = [11.36, 7.89, 1.96, 0, -3.12, -9.52]
  # mean, std = samplemeanstd(dat)
  # z = zscore(dat[0], mean, std)
  # print(z)
  
  #print(zscore(1.853, 1.757, 0.074))
  #print(zscore(1.758, 1.618, 0.069))
  
  # section24(0, 1)

  # mean, std = 150, 8.75
  # print(st.norm.sf(150, mean, std))
  # print(st.norm.cdf(167.5, mean, std) - st.norm.cdf(132.5, mean, std))
  # print(st.norm.sf(159, mean, std))
  # print(st.norm.cdf(170, mean, std) - st.norm.cdf(165, mean, std))

  # mean, std = 150, 8.75
  # print(st.norm.ppf(0.4, mean, std))
  # print(st.norm.isf(0.2, mean, std))

  #print(samplingdistribution([1, 2, 3, 4], 2))
  print(samplingdistribution([5,6,7, 9,13], 2))
  



if(__name__ == "__main__"):
  print("Imports loaded!")
  section24test()
  print("Goodbye!")
  