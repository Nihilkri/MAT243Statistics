from Basic import *

def Zstar(c:float) -> float:
  """ Table 3.2.1: Critical values for common confidence levels. """
  return round(st.norm.ppf(1 - (1 - c) / 2.0), 3)
  confidenceLevels = [0.90, 0.95, 0.99]
  criticalValues = [1.645, 1.960, 2.576]
  if c in confidenceLevels:
    return criticalValues[confidenceLevels.index(c)]
  else:
    print("Confidence level not defined")
  return 0

def Tstar(c:float, df:int) -> float:
  """ Table 3.2.2: Critical values t* for selected degrees of freedom df and confidence level c. """
  return round(st.t.ppf(1 - (1 - c) / 2.0, df), 3)
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
  if std == 0.0:
    # Proportion, not a distribution
    std = (sampleMean * (1 - sampleMean)) ** 0.5
  m = MarginOfError(c, std, n, pop)
  return (sampleMean - m, sampleMean + m)

def GuaranteedSampleSize(c:float, std:float, marginOfError:float, pop:bool, p:float = None) -> int:
  """ Section 3.2.2: Margin of error and sample size for means when popStd is known
      The width of the confidence interval is twice the margin of error. Recall that the margin of
      error depends on the confidence level and the standard error. Thus, given a confidence level,
      the width of the confidence interval changes by changing the standard error. Increasing the
      sample size decreases the standard error. Similarly, decreasing the sample size increases the
      standard error. The size of the sample needed to guarantee a confidence interval with a
      specified margin of error is given by the formula
      n = (Zstar(c) * popStd / marginOfError) ** 2
  """
  if std == 0.0 and p is not None:
    # Section 3.3.2: Margin of error and sample size for proportions
    n = (Zstar(c) / marginOfError) ** 2 * p * (1 - p)
  elif pop:
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

def HypTest(a:float, sampleMean:float, hypPopMean:float, std:float, n:int, tail:int, pop:bool, sig:int=3):
  """ Section 3.5.1 z-test for population means
      A z-test is a hypothesis test in which the z-statistic follows a normal distribution.
      The z-test for a population mean can be used to determine whether the population mean
      is the same as the hypothesized mean mu_0, assuming that the population standard
      deviation sigma is known. When performing a hypothesis test involving the mean of a
      single population with a known population standard deviation, the distribution of the
      z-test statistic is assumed to be N(mu_0, (xbar - mu_0) / (sigma / sqrt(n))). In
      practice, the population standard deviation is rarely known, so a more useful test
      involves the t-distribution because the standard deviation of a sample can always be
      computed.
"""
  if std is None:
    # Procedure 3.6.1: Hypothesis testing for population proportion
    p = sampleMean
    sampleMean = round(sampleMean / n, sig)
    hypPopMean = round(hypPopMean, sig)
    print(f"phat = {p}/{n} = {sampleMean:.6f}")
    print(f"p0 = {hypPopMean:.6f}")
    print(f"{n = }")
    z = round((sampleMean - hypPopMean) / (((hypPopMean * (1 - hypPopMean)) / n) ** 0.5), sig)
  else:
    popStd = std
    z = round((sampleMean - hypPopMean) / (popStd / n ** 0.5), sig)
  if pop:
    # Section 3.5.1 z-test for population means
    print(f"{z = :.6f}")
    if tail == -1:
      p = st.norm.cdf(z, 0, 1)
    elif tail == 1:
      p = st.norm.sf(z, 0, 1)
    else:
      p = st.norm.cdf(-abs(z), 0, 1) + st.norm.sf(abs(z), 0, 1)
  else:
    # Section 3.5.2 t-test for population means
    sampleStd = std
    t = round((sampleMean - hypPopMean) / (sampleStd / n ** 0.5), sig)
    print(f"{t = :.6f}")
    print(f"df = {n - 1}")
    if tail == -1:
      p = st.t.cdf(t, n - 1, 0, 1) 
    elif tail == 1:
      p = st.t.sf(t, n - 1, 0, 1)
    else:
      p = st.t.cdf(-abs(t), n - 1, 0, 1) + st.t.sf(abs(t), n - 1, 0, 1)
  print(f"{p = :.6f}")
  print("Fails to reject null hypothesis" if p > a else "Rejects null hypothesis")




def section3():
  section = "CA3.6.1: Hypothesis test for a population proportion"

  if section == "":
    print(f"{0:.3f}")

  elif section == "CA3.6.1: Hypothesis test for a population proportion":
    q = 3
    if q == 1:
      HypTest(0.01, 120, 0.63, None, 178, 1, True, 3)
    elif q == 2:
      HypTest(0.05, 87, 0.62, None, 125, 1, True, 3)
    elif q == 3:
      HypTest(0.10, 89, 0.77, None, 105, 0, True, 3)


  elif section == "PA3.6.1: Exam scores proportion":
    HypTest(0.01, 31, 0.5, None, 50, 1, True, 30)

  elif section == "Example 3.6.2: Customer satisfaction":
    HypTest(0.05, 132, 0.47, None, 240, 1, True, 3)

  elif section == "Example 3.6.1: Human sex ratio":
    HypTest(0.05, 85, 106/206, None, 189, 0, True, 3)

  elif section == "CA3.5.1: Hypothesis test for a population mean":
    q = 4
    if q == 2:
      HypTest(0.05, 10.06, 10.0, 0.24, 74, 1, True, 2)
    elif q == 3:
      HypTest(0.10, 13.09, 13.0, 0.26, 14, 0, False, 3)
    elif q == 4:
      HypTest(0.05, 20.05, 20.0, 0.25, 33, -1, False, 3)

  elif section == "Example 3.5.3: Circumference of basketballs":
    hypPopMean = 29
    n = 25
    sampleMean = 29.1
    sampleStd = 0.217
    tail = 0
    a = 0.01
    HypTest(a, sampleMean, hypPopMean, sampleStd, n, tail, False, 3)
  
  elif section == "PA3.5.1: Intelligence quotient":
    hypPopMean = 100
    n = 36
    sampleMean = 105
    popStd = 15
    a = 0.05
    tail = 1
    HypTest(a, sampleMean, hypPopMean, popStd, n, tail, True, 3)
  
  elif section == "Example 3.5.2: Carry-on baggage volume":
    hypPopMean = 1.6
    n = 73
    sampleMean = 1.7
    popStd = 0.29
    a = 0.10
    tail = 1
    HypTest(a, sampleMean, hypPopMean, popStd, n, tail, True, 3)
  
  elif section == "Example 3.5.1: Mean battery life":
    hypPopMean = 7.8
    n = 10
    sampleMean = 7.6
    popStd = 0.57
    a = 0.05
    tail = -1
    HypTest(a, sampleMean, hypPopMean, popStd, n, tail, True, 3)
  
  elif section == "CA3.3.1: Confidence intervals for population proportions":
    n = 1000
    p = 618 / n
    print(f"P = {p:.3f}")
    ci = ConfidenceInterval(0.95, p, 0, n, True)
    m = (ci[1] - ci[0]) / 2.0
    print(f"m = {m:.3f}")
    print()
    p = 71 / n
    ci = ConfidenceInterval(0.90, p, 0, n, True)
    print(f"ci = {FormatCI(ci)}")
    print()
    p = 0.31
    m = 0.0393
    c = 0.90
    print(GuaranteedSampleSize(c, 0, m, True, p))
    print()
    m = 0.026
    c = 0.95
    print(GuaranteedSampleSize(c, 0, m, True, 0.5))

  elif section == "PA3.3.2: Margin of error and sample size for proportions":
    p = 0.36
    moe = 0.01
    print(f"1. {(GuaranteedSampleSize(0.90, 0, moe, True, p)):.3f}")
    print(f"2. {(GuaranteedSampleSize(0.95, 0, moe, True, p)):.3f}")
    print(f"3. {math.ceil(GuaranteedSampleSize(0.95, 0, moe, True, 0.50))}")
    print(math.ceil(6234.682))

  elif section == "Example 3.3.2: Six months after Brexit":
    moe = 0.0217
    c = 0.90
    for a, p in [('a', 0.47), ('b', 0.50)]:
      print(f"{a}. {math.ceil(GuaranteedSampleSize(c, 0, moe, True, p))}")
  
  elif section == "PA3.3.1: Confidence interval for the proportion":
    n = 1000
    p = 281 / n
    print(f"1. p = {p:.3f}")
    ci = ConfidenceInterval(0.90, p, 0, n, True)
    print(f"2. moe90 = {(ci[1] - ci[0])/2:.3f}")
    ci = ConfidenceInterval(0.95, p, 0, n, True)
    print(f"3. moe95 = {(ci[1] - ci[0])/2:.3f}")
    print(f"4. CI95 = {FormatCI(ci)}")
    # Python-Function 3.3.1: norm.interval()
    stderr = (p * (1 - p)/n) ** 0.5
    print(FormatCI(st.norm.interval(0.95, p, stderr)))

  elif section == "Example 3.3.1: Confidence interval for the proportion":
    n = 1200
    p = 348 / n
    c = 0.95
    print(f"CI = {FormatCI(ConfidenceInterval(c, p, 0, n, True))}")
  
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

  elif section == "PA3.2.5: Machine output":
    moe = 2.1
    c = 0.90
    sampleStd = 7.3
    print(f"{round(GuaranteedSampleSize(c, sampleStd, moe, True))}")
    print(f"{round(GuaranteedSampleSize(c, sampleStd, moe, False))}")
 
  elif section == "Example 3.2.4: Quality control":
    moe = 0.35
    c = 0.95
    sampleStd = 2.50
    print(f"{math.ceil(GuaranteedSampleSize(c, sampleStd, moe, False))}")

  elif section == "Python-Function 3.2.2: t.interval()":
    n, sampleMean, sampleStd = 100, 219, 35.0
    standardError = sampleStd / (n ** 0.5)
    df = n - 1
    c = 0.95
    ci = st.t.interval(c, df, sampleMean, standardError)
    print(f"{FormatCI(ci)}")
  
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

  elif section == "Example 3.2.3: Circumference of basketballs":
    popMean, n, sampleMean, sampleStd, c = 29, 25, 29.1, 0.217, 0.99
    ci = ConfidenceInterval(c, sampleMean, sampleStd, n, False)
    print(FormatCI(ci))
  
  elif section == "PA3.2.3: Tax assessment":
    sampleMean, popStd = 1400, 1000
    print(f"{math.ceil(GuaranteedSampleSize(0.90, 1000, 100, True))}")

  elif section == "Example 3.2.2: Microbeads in a water reservoir":
    c, marginOfError, popStd = 0.95, 0.8, 2.5
    print(f"{GuaranteedSampleSize(c, popStd, marginOfError, True):0.3f}")

  elif section == "PA3.2.2: Confidence interval":
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

