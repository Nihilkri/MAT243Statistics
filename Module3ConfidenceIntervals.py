from Basic import *
import scipy.stats as st
import Calculus


def Zstar(c:float) -> float:
  """ Table 3.2.1: Critical values for common confidence levels. """
  return st.norm.ppf(1 - (1 - c) / 2.0)
  confidenceLevels = [0.90, 0.95, 0.99]
  criticalValues = [1.645, 1.960, 2.576]
  if c in confidenceLevels:
    return criticalValues[confidenceLevels.index(c)]
  else:
    print("Confidence level not defined")
  return 0

def Tstar(c:float, df:int) -> float:
  """ Table 3.2.2: Critical values t* for selected degrees of freedom df and confidence level c. """
  return st.t.ppf(1 - (1 - c) / 2.0, df)
  confidenceLevels = [0.90, 0.95, 0.99]
  criticalValues = [[2.015, 2.571, 4.032], # 5
                    [1.812, 2.228, 3.169], # 10
                    [1.753, 2.131, 2.947], # 15
                    [1.711, 2.064, 2.797], # 24
                    [1.309, 1.694, 2.449]] # 32
  degreesOfFreedom = [5, 10, 15, 24, 32]
  if c in confidenceLevels:
    if df in degreesOfFreedom:
      return criticalValues[degreesOfFreedom.index(df)][confidenceLevels.index(c)]
    else:
      print(f"Degrees of freedom {df} not defined")
  else:
    print(f"Confidence level {c} not defined")
  return 0

def MarginOfError(c:float, std:float, n:int, pop:bool) -> float:
  """ Section 3.1: Confidence intervals
      A margin of error, denoted by m, is the range of values above and below the point estimate.
      Numerically, m = (critical value)(standard error) where the critical value, which depends
      on c and the underlying distribution of the statistic, is the number of standard errors
      to be added to the point estimate.
  """
  if pop:
    # Section 3.1: Confidence intervals
    popStd = std
    criticalValue = Zstar(c)
    standardError = popStd / (n ** 0.5)
    return criticalValue * standardError
  else:
    # Section 3.2.3: t confidence intervals
    sampleStd = std
    criticalValue = Tstar(c, n - 1)
    standardError = sampleStd / (n ** 0.5)
    return criticalValue * standardError

def ConfidenceInterval(c:float, sampleMean:float, std:float, n:int, pop:bool):
  """ Section 3.1: Confidence intervals
      Two types of estimates exist: point estimates and interval estimates. A point estimate is a
      single value estimate for a parameter. An interval estimate is a range of values that is
      likely to contain the parameter being estimated. Combined with a probability statement, an
      interval estimate is called a confidence interval. The percentage in which the confidence
      interval contains the parameter is called the confidence level, which is denoted by c.
      A confidence interval is constructed by looking at the sample statistic and margin of error.
  """
  m = MarginOfError(c, std, n, pop)
  return (sampleMean - m, sampleMean + m)

def GuaranteedSampleSize(c:float, std:float, marginOfError:float, pop:bool) -> int:
  """ Section 3.2.2: Margin of error and sample size for means when popStd is known
      The width of the confidence interval is twice the margin of error. Recall that the margin of
      error depends on the confidence level and the standard error. Thus, given a confidence level,
      the width of the confidence interval changes by changing the standard error. Increasing the
      sample size decreases the standard error. Similarly, decreasing the sample size increases the
      standard error. The size of the sample needed to guarantee a confidence interval with a
      specified margin of error is given by the formula
      n = (Zstar(c) * popStd / marginOfError) ** 2
  """
  if pop:
    # Section 3.2.2: Margin of error and sample size for means when popStd is known
    popStd = std
    n = (Zstar(c) * popStd / marginOfError) ** 2
  else:
    # Section 3.2.4 Margin of error and sample size for means when popStd is unknown
    sampleStd = std
    nstar = (Zstar(c) * sampleStd / marginOfError) ** 2
    df = int(nstar) - 1
    n = (Tstar(c, df) * sampleStd / marginOfError) ** 2
  return n

def section3():
  section = "CA3.2.1: Confidence intervals for population means"

  if section == "PA3.2.2: Confidence interval":
    dat, popStd = [10, 17, 17.5, 18.5, 19.5], 1.25
    n, sampleMean, sampleStd = MeanStd(dat, True)
    print(f"Sample mean = {sampleMean:.3f}, Pop Std = {popStd:.3f}, n = {n}")
    for c in [0.90, 0.99]:
      print(f"Confidence Level = {c * 100}%")
      standardError = popStd / (n ** 0.5)
      print(f"Standard Error = {standardError:.3f}")
      moe = MarginOfError(c, popStd, n, True)
      print(f"Margin of Error = {moe:.3f}")
      ci = ConfidenceInterval(c, sampleMean, popStd, n, True)
      print(f"Confidence Interval = [{ci[0]:.3f}, {ci[1]:.3f}]")
      # Python-Function 3.2.1: norm.interval()
      ci = st.norm.interval(c, sampleMean, standardError)
      print(f"ST Confidence Interval = [{ci[0]:.3f}, {ci[1]:.3f}]")
      print()

  elif section == "Example 3.2.2: Microbeads in a water reservoir":
    c, marginOfError, popStd = 0.95, 0.8, 2.5
    print(f"{GuaranteedSampleSize(c, popStd, marginOfError, True):0.3f}")

  elif section == "PA3.2.3: Tax assessment":
    sampleMean, popStd = 1400, 1000
    print(f"{math.ceil(GuaranteedSampleSize(0.90, 1000, 100, True))}")

  elif section == "Example 3.2.3: Circumference of basketballs":
    popMean, n, sampleMean, sampleStd, c = 29, 25, 29.1, 0.217, 0.99
    ci = ConfidenceInterval(c, sampleMean, sampleStd, n, False)
    print(FormatCI(ci))
  
  elif section == "PA3.2.4: Weights of pumpkins":
    dat = [5, 7, 7.5, 8, 8.5, 8.75]
    n, sampleMean, sampleStd = MeanStd(dat, True)
    df = n - 1
    c = 0.90
    cv = Tstar(c, df)
    moe = MarginOfError(c, sampleStd, n, False)
    ci = ConfidenceInterval(c, sampleMean, sampleStd, n, False)
    print(f"1. df = {df}")
    print(f"2. cv = {cv:.3f}")
    print(f"3. moe = {moe:.3f}")
    print(f"4. ci = {FormatCI(ci)}")

  elif section == "Python-Function 3.2.2: t.interval()":
    n, sampleMean, sampleStd = 100, 219, 35.0
    standardError = sampleStd / (n ** 0.5)
    df = n - 1
    c = 0.95
    ci = st.t.interval(c, df, sampleMean, standardError)
    print(f"{FormatCI(ci)}")
  
  elif section == "Example 3.2.4: Quality control":
    moe = 0.35
    c = 0.95
    sampleStd = 2.50
    print(f"{math.ceil(GuaranteedSampleSize(c, sampleStd, moe, False))}")

  elif section == "PA3.2.5: Machine output":
    moe = 2.1
    c = 0.90
    sampleStd = 7.3
    print(f"{round(GuaranteedSampleSize(c, sampleStd, moe, True))}")
    print(f"{round(GuaranteedSampleSize(c, sampleStd, moe, False))}")
 
  elif section == "CA3.2.1: Confidence intervals for population means":
    dat = 74, 68, 64, 72, 68, 69, 67, 64, 62
    popStd = 4
    n, sampleMean, sampleStd = MeanStd(dat, True)
    c = 0.90
    print(f"x = {sampleMean:.3f}")
    print(f"Moe = {MarginOfError(c, popStd, n, True):.3f}")
    print(f"CI = {FormatCI(ConfidenceInterval(c, sampleMean, popStd, n, True))}")
    print()

    n, sampleMean, sampleStd = 11, 12, 3
    c = 0.95
    print(f"t* = {Tstar(c, n - 1):.3f}")
    print(f"Moe = {MarginOfError(c, sampleStd, n, False):.3f}")
    print(f"CI = {FormatCI(ConfidenceInterval(c, sampleMean, sampleStd, n, False))}")







  elif section == "":
    print(f"{0}")
  

