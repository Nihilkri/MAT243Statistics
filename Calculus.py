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

def Plot(f, params, r:int, lx:float, rx:float, ly:float, ry:float, title:str, xlabel:str, ylabel:str):
  """ Plots a function F from lx to rx as resolution r """
  xs = np.linspace(lx, rx, r)
  #ys = np.fromfunction(f, (len(xs),))
  ys = f(xs, *params)
  print(xs)
  print(ys)
	
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


def Integrate(f, params, lx:float, rx:float, dx:float) -> float:
  xs = np.linspace(lx, rx, (rx - lx) / dx)
  ys = f(xs, *params)
  return np.sum(ys)



def CalcTest():
  # lambda x: np.sin(x)
  def f(x):
    return np.sin(x)

  print("$$\\func{1}{\\theta}$$")
  #Plot(f, (,), 1001, -4*np.pi, 4*np.pi, "Sine Function", "$\\theta$", "$Sin(\\theta)$")
  Plot(Gaussian, (1.0, 0.71), 10001, -4, 4, None, None, "Gaussian Distribution", "$\\sigma$", "y")