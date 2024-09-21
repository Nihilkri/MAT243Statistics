import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp

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
  fig, ax = plt.subplots()
  if(not (ly is None and ry is None)):
    ax.set_ylim(ly, ry)
  plt.title(title, fontsize=16)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  for xs, ys, c in s:
    plt.plot(xs, ys, c = c)
  plt.grid()
  plt.show()



def Gamma(z:complex) -> complex:
  def f(t:complex, z:complex) -> complex:
    return t ** (z - 1) * np.exp(-t)
  return Integrate(f, (z,), 0, 1000)

def Gaussian(x:float, mean:float, std:float) -> float:
  return (1 / (std*(2 * np.pi) ** 0.5)) * np.exp(-0.5 * ((x - mean) / std) ** 2)

def Students(x:float, mean:float, std:float, df:float) -> float:
  return ((1.0 + x ** 2.0 / df) ** (-(df + 1.0) / 2.0) / (df ** 0.5 * sp.beta(0.5, df / 2.0)))


def Integrate(f, params, lx:float, rx:float, r:int = 1048577) -> float:
  xs, ys = Func(f, params, lx, rx, r)
  dx = (rx - lx) / r
  return np.sum(ys) * dx

def ExpectedValue(f, params, lx:float, rx:float, r:int = 1048577) -> float:
  xs, ys = Func(f, params, lx, rx, r)
  ys *= xs
  dx = (rx - lx) / r
  return np.sum(ys) * dx

def CDF(f, params, lx:float, rx:float, r:int = 1048577) -> float:
  xs, ys = Func(f, params, lx, rx, r)
  dx = (rx - lx) / r
  zs = np.cumsum(ys) * dx
  return xs, zs



def CalcTest():
  if False:
    params, lx, rx, r = None, -4*np.pi, 4*np.pi, 1048577
    f = [(*Func(np.sin, params, lx, rx, r), 'b'),
         (*Func(np.cos, params, lx, rx, r), 'r')]
    Plot(f, "Sine Function", "$\\theta$", "$Sin(\\theta)$", None, None)

  if True:
    params, lx, rx, r = (0.0, 1.0), -4, 4, 1048577
    i = Integrate(Gaussian, params, lx, rx, r)
    print(f"Int_{lx}^{rx} = {i}")
    print(f"Int_{lx}^{rx} **2/pi = {i ** 2 / np.pi}")
    print("Peak =", Gaussian(params[0], *params))
    print()
    e = ExpectedValue(Gaussian, params, lx, rx, r)
    print("E[X] =", e)
    f = [(*CDF(Gaussian, params, lx, rx, r), 'b'),
         (*CDF(Students, (*params, 1), lx, rx, r), 'r'),
         (*CDF(Students, (*params, 2), lx, rx, r), 'g'),
         (*CDF(Students, (*params, 5), lx, rx, r), 'b'),
         (*CDF(Students, (*params, 15), lx, rx, r), 'y')]
    Plot(f, "Gaussian/Students Distribution", "$\\sigma$", "y", None, None)

  if False:
    params, lx, rx, r = None, -4, 4, 1048577
    Plot([(*Func(sp.gamma, params, lx, rx, r), 'b')], "Gamma Function", "x", "$\\Gamma(x)$", -20, 30)
