import math
import scipy.stats as st


def tscore(x:float, n:int, mean:float, std:float):
  """ Section 2.5
      The t-statistic is obtained from a sample assumed to have a t-distribution and involves
      the population mean and a larger variability from estimating the population standard deviation.
      The same process applies to the computation of probabilities involving the t-distribution
      as shown in an earlier section with normal distribution.
  """
  std = std / math.sqrt(n)
  t = (x - mean) / std
  return t, mean, std


def section25test():
  x, n, mean, std, x2 = 2.72, 15, 1.86, 2.12, 3.26
  t, mean1, std1 = tscore(x, n, mean, std)
  t2, mean2, std = tscore(x2, n, mean, std)

  print(f"t = {t}, mean = {mean}, std = {std}")
  print("At most", x)
  print(st.t.cdf(x, n-1, mean, std))
  print(st.t.cdf(t, n-1, 0, 1))
  print("At least", x)
  print(st.t.sf(x, n-1, mean, std))
  print(st.t.sf(t, n-1, 0, 1))
  print("Between", x, "and", x2)
  print(st.t.cdf(x2, n-1, mean, std) - st.t.cdf(x, n-1, mean, std))
  print(st.t.cdf(t2, n-1, 0, 1) - st.t.cdf(t, n-1, 0, 1))

  # Find the 25th percentile of the t-distribution with 30 degrees of freedom. Type as #.###
  print(st.t.ppf(0.25, 30, 0, 1))
  # Find the 60th percentile of the t-distribution with 4 degrees of freedom. Type as #.###
  print(st.t.ppf(0.60, 4, 0, 1))


