from Basic import *


def zstar(c:float) -> float:
  if(c == 0.90):
    return 1.645
  elif c == 0.95:
    return 1.960
  elif c == 0.99:
    return 2.576
  else:
    print("Confidence level not defined")
  return 0

def marginoferror(samplemean: float, samplestd:float, n:int, c:float):
  return zstar(c) * samplestd / (n ** 0.5)

def confidenceinterval(samplemean: float, samplestd:float, n:int, c:float):
  m = marginoferror(samplemean, samplestd, n, c)
  return (samplemean - m, samplemean + m)


def section31test():
  dat = [10, 17, 17.5, 18.5, 19.5]
  popstd = 1.25
  n, samplemean, samplestd = meanstd(dat, True, 3)
  print(f"Sample mean = {samplemean:.3f}, Pop Std = {popstd:.3f}, n = {n}")
  for c in [0.90, 0.99]:
    print(f"Confidence Level = {c * 100}%")
    moe = marginoferror(samplemean, popstd, n, c, 30)
    print(f"Margin of Error = {moe:.3f}")
    ci = confidenceinterval(samplemean, popstd, n, c, 30)
    print(f"Confidence Interval = [{ci[0]:.3f}, {ci[1]:.3f}]")
    print()
  

