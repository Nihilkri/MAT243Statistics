import math
import numpy as np
#import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
#import scikit-learn as sks
#import seaborn as sns
#import quandl as qn

def plotSine():
	x = [th/10000.0 for th in range(math.ceil(2*np.pi*10000))]
	y = [np.sin(th) for th in x]

	plt.plot(x, y)
	plt.show()

if(__name__ == "__main__"):
	print("Imports loaded!")
	plotSine()
	print("Goodbye!")
