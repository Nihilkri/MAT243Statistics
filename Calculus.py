import math
import numpy as np
import matplotlib.pyplot as plt

theta = chr(0x03B8)
sigma = chr(0x3C3)


def Plot(f, r:int, lx:float, rx:float, title:str, xlabel:str, ylabel:str):
  """ Plots a function F from lx to rx as resolution r """
  xs = np.linspace(lx, rx, r)
  #ys = np.fromfunction(f, (len(xs),))
  ys = f(xs)
  print(xs)
  print(ys)
	
  plt.title(title, fontsize=16)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.plot(xs, ys)
  plt.show()





def plotSine():
  """ Plots a basic sine wave """
  x = [th/10000.0 for th in range(math.ceil(2*np.pi*10000))]
  y = [np.sin(th) for th in x]
  
  plt.title("Sine function", fontsize=16)
  plt.xlabel(theta)
  plt.ylabel(f"sin({theta})")
  plt.plot(x, y)
  plt.show()



def CalcTest():
  # lambda x: np.sin(x)
  def f(x):
    return np.sin(x)

  Plot(f, 1001, -4*np.pi, 4*np.pi, "Sine Function", "$\\theta$", "$Sin(\\theta)$")
  print("$$\\func{1}{\\theta}$$")
