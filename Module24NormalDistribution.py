from Basic import *
import scipy.stats as st


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
  popmean, popstd = MeanStd(dat, False)
  popstd = round(popstd, 2)
  samples = allsamples(dat, sampn)
  numsamples = len(samples)
  samplemeans = [sum(sample) / sampn for sample in samples]
  meansamplemeans, samplestd = MeanStd(samplemeans)
  SE = popstd / math.sqrt(sampn)
  correctionfactor = math.sqrt((popn - sampn) / (popn - 1))
  distrostd = SE * correctionfactor
  print("dat =", dat)
  print("popmean =", popmean)
  print("popstd =", popstd)
  print("samples =", samples)
  print("samplemeans =", samplemeans)
  print("meansamplemeans =", meansamplemeans)
  print("samplestd =", samplestd)
  print("correctionfactor =", correctionfactor)
  print("SE =", SE)
  print("distrostd =", distrostd)

  return samples

def zscore(x:float, mean:float, std:float) -> float:
  """ 2.4 A z-score is a signed value that indicates the number of 
      standard deviations a quantity is from the mean
  """
  return (x - mean) / std

def CLT(x:float, dmean:float, dstd:float, sampn:int, popn:int, r:int = 3):
  """ Central Limit Theorem
      Randomness assumption - samples must be randomly selected.
      Independence condition - sample values must be independent from each other.
      Sample size assumption - sample size must be large enough.
      A rule of thumb is that sample sizes should be at least 30.
      10% condition - sample size must be at most 10% of the population size.
  """
  mean = round(dmean, r)
  std = round(dstd / math.sqrt(sampn), r)
  z = round(zscore(x, mean, std), r)
  return z, mean, std

def binomialdistribution(phat:float, n:int, p:float, r:int = 3):
  """ The Central Limit Theorem for proportions
      It states that if X ~ B(n, p) where n is the number of trials and
      p is the probability of success, then the sampling distribution for
      proportions phat follows a normal distribution N(p, sqrt((p*(1-p))/n)).
  """
  if(n * p < 5 or n * (1 - p) < 5): return 0
  mean = round(p, r)
  std = round(math.sqrt(p * (1 - p) / n), r)
  z = round(zscore(phat, mean, std), r)
  return z, mean, std


def normalDistributionDefinition(mean:float, std:float):
  """ 2.4 Normal Distribution """
  for i in range(1, 6):
    lval = st.norm.cdf(-i, mean, std)
    rval = st.norm.cdf(i, mean, std)
    lrval = (rval - lval) * 100
    lstr = f"{-i}{sigma}={mean - i * std}"
    rstr = f"{i}{sigma}={mean + i * std}"
    print(f"{lrval:.6f}% of samples fall between {lstr} and {rstr}")

  print()
  for i in range(1, 6):
    lval = st.norm.cdf(i - 1, mean, std)
    rval = st.norm.cdf(i, mean, std)
    lrval = (rval - lval) * 100
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
  # 2.4.2: Empirical rule
  #normalDistributionDefinition(511, 120)
  
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
  #print(samplingdistribution([5,6,7, 9,13], 4))

  x, sampn, popn = 180, 36, 1
  mean, std = 172, 29
  z, mean, std = CLT(x, mean, std, sampn, popn)

  # x, n, p = 0.09, 256, 0.08
  # z, mean, std = binomialdistribution(x, n, p)

  print(f"z = {z}, mean = {mean}, std = {std}")
  print("At most", x)
  print(st.norm.cdf(x, mean, std))
  print(st.norm.cdf(z, 0, 1))
  print("At least", x)
  print(st.norm.sf(x, mean, std))
  print(st.norm.sf(z, 0, 1))

