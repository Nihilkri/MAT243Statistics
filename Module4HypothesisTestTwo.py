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


def section4():
  section = "Python-Function 4.1.1: ztest(x1, x2)"

  if section == "":
    print(f"{0:.3f}")

  elif section == "":
    print(f"{0:.3f}")

  elif section == "":
    print(f"{0:.3f}")

  elif section == "":
    print(f"{0:.3f}")

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