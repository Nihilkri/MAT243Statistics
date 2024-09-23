#from Basic import *
import scipy.stats as st
#import Calculus

def Ztest2(sampleMean1:float, sampleMean2:float, std1:float, std2:float, n1:int, n2:int, tail:int, a:float, sig:int=3):
  """ Section 4.1.1 Two-sample z-test for population means
      The z-test can also be used to determine whether the means of two independent populations are
      the same when the population standard deviations are known. When performing a hypothesis test
      involving the means of two independent populations, the distribution of the z-test statistic
      is assumed to be N(0, sqrt(sigma_1^2 / n_1 + sigma_2^2 / n_2)). In practice, the standard
      deviation for populations are generally unknown, so either the paired or unpaired t-test
      is needed.
      The conditions that must be satisfied are similar to those of a z-test for a population mean.
  """
  popStd1, popStd2 = std1, std2
  stderr = round((popStd1 ** 2 / n1 + popStd2 ** 2 / n2) ** 0.5, sig)
  print(f"{stderr = :.6f}")
  z = round((sampleMean1 - sampleMean2 - 0) / stderr, sig)
  print(f"{z = :.6f}")
  if tail == -1:
    p = st.norm.cdf(z, 0, 1)
  elif tail == 1:
    p = st.norm.sf(z, 0, 1)
  else:
    p = st.norm.cdf(-abs(z), 0, 1) + st.norm.sf(abs(z), 0, 1)
  print(f"{p = :.6f}")
  print("Fails to reject null hypothesis" if p > a else "Rejects null hypothesis")
  
def pairedTTest(sampleMeanDifference:float, hypMeanDifference:float, sampleStd:float, n:int, tail:int, a:float, sig:int=3):
  """ Section 4.1.2 Two-sample t-test
       In a paired t-test or dependent t-test, a sample taken from one population is exposed to two different treatments. The main idea is that measurements are recorded from the same group, usually before and after a treatment is applied or when each of two treatments is applied. Ex: A group of professional cycling athletes is selected for a study on the effects of caffeine dosage on exhaustion times. The populations are the cyclists for each of two dosages. The samples are the measured exhaustion times for each dosage, which implies dependence because the measurements were taken from the same group.
  """
  t = round((sampleMeanDifference - hypMeanDifference) / (sampleStd / n ** 0.5), sig)
  t = 1.418
  print(f"{t = :.6f}")
  df = n - 1
  if tail == -1:
    p = st.t.cdf(t, df, 0, 1)
  elif tail == 1:
    p = st.t.sf(t, df, 0, 1)
  else:
    p = st.t.cdf(-abs(t), df, 0, 1) + st.t.sf(abs(t), df, 0, 1)
  print(f"{p = :.6f}")
  print("Fails to reject null hypothesis" if p > a else "Rejects null hypothesis")
  
  
def UnpairedTTest():
  """ Section 4.1.2 Two-sample t-test
       In an unpaired t-test or independent t-test, a sample taken from one population is not related to a different sample taken from another population. In contrast to the paired t-test, measurements from an unpaired t-test are recorded from different groups when exposed to the same treatment. Ex: The effect of caffeine intake on exhaustion times is studied by measuring the exhaustion times of a randomly selected group of 9 professional cyclists taking caffeine pills and another group of 9 cyclists not taking caffeine pills. The two populations are all cyclists taking caffeine pills and those who are not taking the pills. The samples are the measured exhaustion times from the two groups, each with 9 cyclists, which implies independence because the times are for two different groups of cyclists.
  """
  pass
        
        


def section4():
  section = "zyDE 4.1.1: Paired t-test"

  if section == "":
    print(f"{0:.3f}")

  elif section == "":
    print(f"{0:.3f}")

  elif section == "Example 4.1.3: Improvement in exam scores":
    print(f"{0:.3f}")

  elif section == "zyDE 4.1.1: Paired t-test":
    """ The st.ttest_rel(x, y) function takes two arrays or DataFrame columns x and y with the same length as inputs and returns the t-statistic and the corresponding two-tailed p-value as outputs.

      The code below loads the exam scores data and uses a paired t-test for the hypotheses H_0 : mu_d = 0 and H_a : mu_d != 0 where mu_d is the average difference in students' exam 1 and exam 2 scores.
    """
    import pandas as pd
    df = pd.read_csv('ExamScores.csv')
    print(st.ttest_rel(df['Exam1'],df['Exam2']))

  elif section == "Python-Function 4.1.1: ztest(x1, x2)":
    #from statsmodels.stats.weightstats import ztest
    #sample1 = [21, 28, 40, 55, 58, 60]
    #sample2 = [13, 29, 50, 55, 71, 90]
    #print(ztest(x1 = sample1, x2 = sample2))
    pass

  elif section == "PA4.1.1: Commute times":
    Ztest2(5.35, 4.95, 0.5, 0.8, 40, 50, 1, 0.05, 4)

  elif section == "Example 4.1.2: Extrasensory perception":
    Ztest2(12, 10, 4.5, 4.0, 50, 60, 0, 0.05)

  elif section == "Example 4.1.1: Candle burn times":
    Ztest2(27.5, 26.0, 2.5, 3.5, 100, 100, 1, 0.05)


if(__name__ == "__main__"):
  print("Imports loaded!")
  section4()
  print("Goodbye!")