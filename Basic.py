print("Loading imports")
import math
print("Math loaded")
import numpy as np
print("Numpy loaded")
import scipy.stats as st
print("Scipy.stats loaded")
import pandas as pd
print("Pandas loaded")
import matplotlib.pyplot as plt
plt.style.use('dark_background')
print("Pyplot loaded")
# import sklearn as sks
# #print("Scikit-learn loaded")
import seaborn as sns
print("Seaborn loaded")
# import quandl as qn
# print("Imports loaded!\n")


# Greek Letters
beta    = "\u03B2"
epsilon = "\u03B5"
theta   = "\u03B8"
mu      = "\u03BC"
sigma   = "\u03C3"
Sigma   = "\u03A3"
# Superscripts
squared = "\u00B2"
# Subscripts
subi    = "\u1D62"
subn    = [chr(0x2080+i) for i in range(10)]
#subnn   = ["\u208" + str(i) + "\u208" + str(j) for j in range(10) for i in range(10)]
# Diacritics
hat     = "\u0302"
bar     = "\u0304"
# Control / Special
frac    = "\u2044"
ne      = "\u2260"

def abandc(abc:list[str], noand:bool = False) -> str:
  """ Converts [A, B, C] into 'A, B, and C' """
  n:int = len(abc)
  if(n == 0): return s
  s:str = str(abc[0])
  if(n == 1): return s
  for a in range(1, n - 1):
    s += ", " + str(abc[a])
  s += ("," if n > 2 else "") + (" " if noand else " and ") + str(abc[-1])
  return s

def MeanStd(dat:list[float], sample:bool = True) -> tuple[int, float, float]:
  """ Calculates the mean and standard deviation of either the population or a sample.
      The mean is from section 1.12 and the standard deviation is from section 1.13.
  """
  n = len(dat)
  mean = sum(dat) / n
  sumsqd = sum([(x - mean) ** 2 for x in dat])
  smpstd = math.sqrt(sumsqd / (n - (1 if sample else 0)))
  return n, mean, smpstd

def ZTest(z:float, tail:int, a:float, sig:int=3):
  print(f"{z = :.{sig}f}")
  if tail == 0:
    p = st.norm.cdf(-abs(z), 0, 1) + st.norm.sf(abs(z), 0, 1)
  else:
    p = st.norm.cdf(-tail * z, 0, 1)
  print(f"{p = :.{sig}f}")
  print("Fails to reject null hypothesis" if p > a else "Rejects null hypothesis")

def TTest(t:float, df:int, tail:int, a:float, sig:int=3):
  print(f"{t = :.{sig}f}")
  print(f"{df = }")
  if tail == 0:
    p = st.t.cdf(-abs(t), df, 0, 1) + st.t.sf(abs(t), df, 0, 1)
  else:
    p = st.t.cdf(-tail * t, df, 0, 1)
  print(f"{p = :.{sig}f}")
  print("Fails to reject null hypothesis" if p > a else "Rejects null hypothesis")

def stats(dat:list[float]):
  """ The list of statistics on a dataset, from sections 1.12 and 1.13 """
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
    print(f"Mode{('s' if len(modes) > 1 else '')}: {abandc(modes)}, with count {modecount}")

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

def FormatCI(ci:tuple[float]) -> str:
  return f"[{ci[0]:.3f}, {ci[1]:.3f}]"

from typing import Generator
import colorsys
def colorWheel(n:int) -> Generator[None, str, None]:
  """
  """