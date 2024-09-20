from Basic import *
import scipy.stats as st


def Zstar(c:float) -> float:
  """ Table 3.2.1: Critical values for common confidence levels. """
  if(c == 0.90):
    return 1.645
  elif c == 0.95:
    return 1.960
  elif c == 0.99:
    return 2.576
  else:
    print("Confidence level not defined")
  return 0

def MarginOfError(sampleMean: float, popStd:float, n:int, c:float):
  """ Section 3.1 Confidence intervals
      A margin of error, denoted by m, is the range of values above and below the point estimate.
      Numerically, m = (critical value)(standard error) where the critical value, which depends
      on c and the underlying distribution of the statistic, is the number of standard errors
      to be added to the point estimate.
  """
  criticalValue = Zstar(c)
  standardError = popStd / (n ** 0.5)
  return criticalValue * standardError

def ConfidenceInterval(sampleMean: float, popStd:float, n:int, c:float):
  """ Section 3.1 Confidence intervals
      Two types of estimates exist: point estimates and interval estimates. A point estimate is a
      single value estimate for a parameter. An interval estimate is a range of values that is
      likely to contain the parameter being estimated. Combined with a probability statement, an
      interval estimate is called a confidence interval. The percentage in which the confidence
      interval contains the parameter is called the confidence level, which is denoted by c.
      A confidence interval is constructed by looking at the sample statistic and margin of error.
  """
  m = MarginOfError(sampleMean, popStd, n, c)
  return (sampleMean - m, sampleMean + m)


def section3test():
  section = "Example 3.2.2: Microbeads in a water reservoir"

  if section == "PA3.2.2: Confidence interval":
    dat, popStd = [10, 17, 17.5, 18.5, 19.5], 1.25

    n, sampleMean, sampleStd = MeanStd(dat, True)
    print(f"Sample mean = {sampleMean:.3f}, Pop Std = {popStd:.3f}, n = {n}")
    for c in [0.90, 0.99]:
      print(f"Confidence Level = {c * 100}%")
      standardError = popStd / (n ** 0.5)
      print(f"Standard Error = {standardError:.3f}")
      moe = MarginOfError(sampleMean, popStd, n, c)
      print(f"Margin of Error = {moe:.3f}")
      ci = ConfidenceInterval(sampleMean, popStd, n, c)
      print(f"Confidence Interval = [{ci[0]:.3f}, {ci[1]:.3f}]")
      # Python-Function 3.2.1: norm.interval()
      ci = st.norm.interval(c, sampleMean, standardError)
      print(f"ST Confidence Interval = [{ci[0]:.3f}, {ci[1]:.3f}]")
      print()

  if section == "Example 3.2.2: Microbeads in a water reservoir":
    pass
  

