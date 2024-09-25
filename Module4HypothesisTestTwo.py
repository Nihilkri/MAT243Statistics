#from Basic import *
from Module3ConfidenceIntervals import ZTest, TTest
import scipy.stats as st
#import Calculus

def ZTest2(sampleMean1:float, sampleMean2:float, popStd1:float, popStd2:float, n1:int, n2:int, tail:int, a:float, sig:int=3):
  """ Section 4.1.1 Two-sample z-test for population means
      The z-test can also be used to determine whether the means of two independent populations are
      the same when the population standard deviations are known. When performing a hypothesis test
      involving the means of two independent populations, the distribution of the z-test statistic
      is assumed to be N(0, sqrt(sigma_1^2 / n_1 + sigma_2^2 / n_2)). In practice, the standard
      deviation for populations are generally unknown, so either the paired or unpaired t-test
      is needed.
      The conditions that must be satisfied are similar to those of a z-test for a population mean.
  """
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
  
def PairedTTest(sampleMeanDifference:float, hypMeanDifference:float, sampleStd:float, n:int, tail:int, a:float, sig:int=3):
  """ Section 4.1.2 Two-sample t-test
       In a paired t-test or dependent t-test, a sample taken from one population is exposed to two different treatments. The main idea is that measurements are recorded from the same group, usually before and after a treatment is applied or when each of two treatments is applied. Ex: A group of professional cycling athletes is selected for a study on the effects of caffeine dosage on exhaustion times. The populations are the cyclists for each of two dosages. The samples are the measured exhaustion times for each dosage, which implies dependence because the measurements were taken from the same group.
  """
  t = round((sampleMeanDifference - hypMeanDifference) / (sampleStd / n ** 0.5), sig)
  print(f"{t = :.6f}")
  df = n - 1
  print(f"{df = }")
  if tail == -1:
    p = st.t.cdf(t, df, 0, 1)
  elif tail == 1:
    p = st.t.sf(t, df, 0, 1)
  else:
    p = st.t.cdf(-abs(t), df, 0, 1) + st.t.sf(abs(t), df, 0, 1)
  print(f"{p = :.6f}")
  print("Fails to reject null hypothesis" if p > a else "Rejects null hypothesis")
  
  
def UnpairedTTest(sampleMean1:float, sampleMean2:float, hypSampleMean1:float, hypSampleMean2:float, sampleStd1:float, sampleStd2:float, n1:int, n2:int, tail:int, a:float, sig:int=3):
  """ Section 4.1.2 Two-sample t-test
       In an unpaired t-test or independent t-test, a sample taken from one population is not related to a different sample taken from another population. In contrast to the paired t-test, measurements from an unpaired t-test are recorded from different groups when exposed to the same treatment. Ex: The effect of caffeine intake on exhaustion times is studied by measuring the exhaustion times of a randomly selected group of 9 professional cyclists taking caffeine pills and another group of 9 cyclists not taking caffeine pills. The two populations are all cyclists taking caffeine pills and those who are not taking the pills. The samples are the measured exhaustion times from the two groups, each with 9 cyclists, which implies independence because the times are for two different groups of cyclists.
  """
  stderr = round((sampleStd1 ** 2 / n1 + sampleStd2 ** 2 / n2) ** 0.5, sig)
  print(f"{stderr = :.6f}")
  t = round((sampleMean1 - sampleMean2 - (hypSampleMean1 - hypSampleMean2)) / stderr, sig)
  print(f"{t = :.6f}")
  df = n1 + n2 - 2
  print(f"{df = }")
  if tail == -1:
    p = st.t.cdf(t, df, 0, 1)
  elif tail == 1:
    p = st.t.sf(t, df, 0, 1)
  else:
    p = st.t.cdf(-abs(t), df, 0, 1) + st.t.sf(abs(t), df, 0, 1)
  print(f"{p = :.6f}")
  print("Fails to reject null hypothesis" if p > a else "Rejects null hypothesis")

def ZTest2Prop(samp1:float, n1:int, samp2:float, n2:int, tail:int, a:float, sig:int=3):
  """ Section 4.2.1 Hypothesis test for the difference between two population proportions
      The z-test can also be used to determine whether the proportions of two distinct populations
      are the same. 
      Where p1 is the probability of success in the first sample, p2 is the probability of success
      in the second sample, p is the overall probability of success when two samples are combined,
      n1 is the size of the first sample, and n2 is the size of the second sample.
  """
  p1 = round(samp1 / n1, sig)
  p2 = round(samp2 / n2, sig)
  p = round((samp1 + samp2) / (n1 + n2), sig)
  if(samp1 < 5):
    print(f"The first proportion is too small. {n1}*{p1} < 5.")
    return 0
  if(n1 - samp1 < 5):
    print(f"The first proportion is too large. {n1}*{p1} > {n1} - 5.")
    return 0
  if(samp2 < 5):
    print(f"The second proportion is too small. {n2}*{p2} < 5.")
    return 0
  if(n2 - samp2 < 5):
    print(f"The second proportion is too large. {n2}*{p2} > {n2} - 5.")
    return 0
  print(f"{p1 = :.{sig}f}")
  print(f"{p2 = :.{sig}f}")
  print(f"p^ = {p:.{sig}f}")
  z = round((p1 - p2 - 0) / (p * (1 - p) * (1 / n1 + 1 / n2)) ** 0.5, sig)
  ZTest(z, tail, a, sig)
        
#==================================================================================================


def Section4():
  section = "Example 4.2.1: Gender and voting"

  if section == "":
    print(f"{0:.3f}")

  elif section == "":
    print(f"{0:.3f}")

  elif section == "":
    print(f"{0:.3f}")

  elif section == "":
    print(f"{0:.3f}")

  elif section == "Example 4.2.1: Gender and voting":
    ZTest2Prop(70, 132, 63, 105, 0, 0.05, 3)

  elif section == "normalDistributionDefinition":
    from Module24NormalDistribution import normalDistributionDefinition as gauss
    gauss(0, 1)

  elif section == "CA4.1.1: Hypothesis test for the difference between two population means":
    #UnpairedTTest(130, 126, 0, 0, 5, 3, 12, 12, 0, 0.01, 3)
    PairedTTest(84-81, 0, 6, 11, 0, 0.01, 3)

  elif section == "zyDE 4.1.2: Unpaired t-test":
    """ The st.ttest_ind(x, y) command takes two arrays or DataFrame columns x and y as inputs and
        returns the t-statistic and the corresponding two-tailed p-value as outputs.
        The code below conducts the test for the packing machine example above. Both the two-tailed
        test and the one-tailed test are conducted.
    """
    import pandas as pd
    df = pd.read_csv('Machine.csv')
    # Two-tailed test
    print(st.ttest_ind(df['Old'],df['New'],equal_var=False, alternative="two-sided"))
    # One-tailed test
    print(st.ttest_ind(df['Old'],df['New'],equal_var=False, alternative="greater"))
    
  elif section == "zyDE 4.1.1: Paired t-test":
    """ The st.ttest_rel(x, y) function takes two arrays or DataFrame columns x and y with the same
        length as inputs and returns the t-statistic and the corresponding two-tailed p-value as
        outputs.
        The code below loads the exam scores data and uses a paired t-test for the hypotheses
        H_0 : mu_d = 0 and H_a : mu_d != 0 where mu_d is the average difference in students'
        exam 1 and exam 2 scores.
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
    ZTest2(5.35, 4.95, 0.5, 0.8, 40, 50, 1, 0.05, 4)

  elif section == "Example 4.1.2: Extrasensory perception":
    ZTest2(12, 10, 4.5, 4.0, 50, 60, 0, 0.05)

  elif section == "Example 4.1.1: Candle burn times":
    ZTest2(27.5, 26.0, 2.5, 3.5, 100, 100, 1, 0.05)


if(__name__ == "__main__"):
  print("Imports loaded!")
  Section4()
  print("Goodbye!")