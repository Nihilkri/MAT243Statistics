import math
import numpy as np
import matplotlib.pyplot as plt

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

def Plot(f, params, title:str, xlabel:str, ylabel:str, lx:float, rx:float, r:int = 1048577, ly:float = None, ry:float = None):
  """ Plots a function F from lx to rx as resolution r """
  xs = np.linspace(lx, rx, r)
  #ys = np.fromfunction(f, (len(xs),))
  ys = f(xs, *params)
  #print(xs)
  #print(ys)
	
  fig, ax = plt.subplots()
  if(not (ly is None and ry is None)):
    ax.set_ylim(ly, ry)
  plt.title(title, fontsize=16)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.plot(xs, ys)
  plt.grid()
  plt.show()

def Gaussian(x:np.ndarray, mean:float, std:float) -> float:
  return (1 / (std*(2 * np.pi) ** 0.5)) * np.exp(-0.5 * ((x - mean) / std) ** 2)


def Integrate(f, params, lx:float, rx:float, r:int = 1048577) -> float:
  xs = np.linspace(lx, rx, r)
  ys = f(xs, *params)
  dx = (rx - lx) / r
  return np.sum(ys) * dx



def CalcTest():
  # lambda x: np.sin(x)
  def f(x):
    return np.sin(x)
  params, lx, rx, r = (0.0, 1.0), -3, 3, 1048577
  i = Integrate(Gaussian, params, lx, rx, r)
  print(f"Int_{lx}^{rx} = {i}")
  print(f"Int_{lx}^{rx} **2/pi = {i ** 2 / np.pi}")
  print("Peak =", Gaussian(params[0], *params))
  print()
  
  print("2**20+1 =", 2**20+1)
  #Plot(f, (,), "Sine Function", "$\\theta$", "$Sin(\\theta)$", -4*np.pi, 4*np.pi, 1048577)
  #Plot(Gaussian, params, "Gaussian Distribution", "$\\sigma$", "y", lx, rx, r, None, None)
