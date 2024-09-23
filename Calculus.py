import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import random

theta = chr(0x03B8)
sigma = chr(0x3C3)


def plotSine():
  """ Plots a basic sine wave """
  x = [th/10000.0 for th in range(math.ceil(2*np.pi*10000))]
  y = [np.sin(th) for th in x]
  
  plt.title("Sine function", fontsize=16)
  plt.xlabel(theta)
  plt.ylabel(f"sin({theta})")
  plt.plot(x, y)
  plt.show()

def Func(f, params, lx:float, rx:float, r:int = 1048577):
  """ Defines a function F from lx to rx as resolution r """
  xs = np.linspace(lx, rx, r)
  if params is None:
    ys = f(xs)
  else:
    ys = f(xs, *params)
  return xs, ys

def Plot(s, title:str, xlabel:str, ylabel:str, ly:float = None, ry:float = None):
  """ Plots a function F from lx to rx as resolution r """
  plt.style.use('dark_background')
  fig, ax = plt.subplots()
  #ax.set_yticks(np.linspace(0, 1, 11))
  #ax.set_yticks(np.linspace(0, 1, 101), minor=True)
  if(not (ly is None and ry is None)):
    ax.set_ylim(ly, ry)
  plt.title(title, fontsize=16)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  for xs, ys, c, dbug in s:
    plt.plot(xs, ys, c = c)
    if dbug:
      print("xs ", (xs * 100).tolist())
      print("ys ", (ys * 100).tolist())
  plt.grid()
  plt.show()



def Gamma(z:complex) -> complex:
  def f(t:complex, z:complex) -> complex:
    return t ** (z - 1) * np.exp(-t)
  return Integrate(f, (z,), 0, 1000)

def Gaussian(x:np.ndarray, mean:float, std:float, skl:float) -> np.ndarray:
  return (skl / (std*(2 * np.pi) ** 0.5)) * np.exp(-0.5 * ((x - mean) / std) ** 2)

def Students(x:np.ndarray, mean:float, std:float, df:float) -> np.ndarray:
  return ((1.0 + x ** 2.0 / df) ** (-(df + 1.0) / 2.0) / (df ** 0.5 * sp.beta(0.5, df / 2.0)))

def Const(x:np.ndarray) -> np.ndarray:
  return np.linspace(0.125, 0.125, len(x))

def Dice(xs:np.ndarray, n:int, d:int) -> np.ndarray:
  lxs = len(xs)
  lnd = n*d
  yscale = lnd / lxs
  distr = np.zeros(lnd)
  samples = np.zeros(lxs)
  for sample in range(lxs):
    y = 0
    for roll in range(n):
      y += random.randint(1, d)
    samples[sample] = y
    distr[y] += 1
  ys = np.zeros(lxs)
  for i in range(lxs):
    ys[i] = distr[int(i*lnd/lxs)]
  ys *= yscale
  ssamples = sorted(samples)
  return samples, ssamples, ys




def Integrate(f, params, lx:float, rx:float, r:int = 1048577) -> np.ndarray:
  xs, ys = Func(f, params, lx, rx, r)
  dx = (rx - lx) / r
  return np.sum(ys) * dx

def ExpectedValue(f, params, lx:float, rx:float, r:int = 1048577) -> np.ndarray:
  xs, ys = Func(f, params, lx, rx, r)
  ys *= xs
  dx = (rx - lx) / r
  return np.sum(ys) * dx

def CDF(f, params, lx:float, rx:float, r:int = 1048577) -> np.ndarray:
  xs, ys = Func(f, params, lx, rx, r)
  dx = (rx - lx) / r
  zs = np.cumsum(ys) * dx
  return xs, zs



def CalcTest():
  q = 4
  if q == 0:
    params, lx, rx, r = None, -4*np.pi, 4*np.pi, 1048577
    f = [(*Func(np.sin, params, lx, rx, r), 'b'),
         (*Func(np.cos, params, lx, rx, r), 'r')]
    Plot(f, "Sine Function", "$\\theta$", "$Sin(\\theta)$", None, None)

  elif q == 1:
    params, lx, rx, r = (0.0, 1.0, 1.0), -4, 4, 1048577
    # i = Integrate(Gaussian, params, lx, rx, r)
    # print(f"Int_{lx}^{rx} = {i}")
    # print(f"Int_{lx}^{rx} **2/pi = {i ** 2 / np.pi}")
    # print("Peak =", Gaussian(params[0], *params))
    # print()
    # e = ExpectedValue(Gaussian, params, lx, rx, r)
    # print("E[X] =", e)
    f = [(*Func(Gaussian, params, lx, rx, 9), 'b', True),
         (*CDF(Gaussian, params, lx, rx, 9), 'r', True),
         (*Func(Gaussian, params, lx, rx, r), 'g', False),
         (*CDF(Gaussian, params, lx, rx, r), '#FF00FF', False),
         (*Func(Const, None, lx, rx, r), '#FF8000', False),
         (*CDF(Const, None, lx, rx, r), '#FFFFFF', False)]
    Plot(f, "Gaussian/Students Distribution", "$\\sigma$", "y", None, None)

  elif q == 2:
    params, lx, rx, r = None, -4, 4, 1048577
    Plot([(*Func(sp.gamma, params, lx, rx, r), 'b')], "Gamma Function", "x", "$\\Gamma(x)$", -20, 30)

  elif q == 3:
    n, d = 8, 6
    nd = n * d
    die = np.linspace(1, d, d)
    print(die)
    mean = die.mean() * n
    std = die.std() * n ** 0.5
    print(mean, std)
    params, lx, rx, r = (mean, std, nd), 0, nd, 10001#1048577
    xs = np.linspace(lx, rx, r)
    samples, ssamples, ys = Dice(xs, n, d)
    gaussian = Gaussian(xs, *params)
    cdf = np.cumsum(gaussian)*nd/r
    print(gaussian.max())
    f = [(xs, samples, 'r', False),
         (xs, ssamples, 'g', False),
         (xs, ys, 'b', False),
         (xs, gaussian, '#FFFFFF', False),
         (xs, cdf, '#FF8000', False)]
    Plot(f, f"{n}d{d} Distribution", "Value of the roll", "Number of rolls", None, None)
  
  elif q == 4:
    params, lx, rx, r = None, -4*np.pi, 4*np.pi, 1048577
    xs = np.linspace(lx, rx, r)
    ys1 = np.sin(xs) + 51.0
    ys2 = np.sin(xs) + 49.0
    ys = (ys1 + ys2) / 2.0
    f = [(xs, ys1, 'r', False),
         (xs, ys2, 'b', False),
         (xs, ys, 'w', False)]
    Plot(f, "Sine Function", "$\\theta$", "$Sin(\\theta)$", None, None)
