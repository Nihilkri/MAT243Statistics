import math
import scipy.stats as st


def section22(x:list, p:list):
  """ 2.2 Properties of discrete probability distributions
  x defines a list containing the outcomes in the sample space
  p defines a list containing the probabilities for each outcome """
    
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

