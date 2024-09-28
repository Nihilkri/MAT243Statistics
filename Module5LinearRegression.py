from Basic import *
import scipy.stats as st
import pandas as pd


def FormatLinregress(inp):
  slope, intercept, r, p, se = inp
  istderr = inp.intercept_stderr
  names = ['Slope', 'Intercept', 'R Value', 'P Value', 'Standard Error', 'Intercept Standard Error']
  values = [slope, intercept, r, p, se, istderr]
  return {'Names':names, 'Values':values}

def SimpleLinearRegression(x:np.ndarray, y:np.ndarray, b0:float, b1:float):
  """ Section 5.1.1 Regression Lines
        A simple linear regression is a way to model the linear relationship between two
        quantitative variables, using a line drawn through those variables' data points,
        known as a regression line.
      Section 5.1.5 Least squares method
        Summing the absolute errors is one method to measure how far the line is from the points.
        Another method to measure error is by computing the sum of squared errors. The sum of
        squared errors is the sum of the differences between the Y values of the data points and
        the values obtained from the regression line.
  """
  lx, ly = len(x), len(y)
  if lx != ly:
    print("X Y mismatched!")
    return None
  ey = b0 + b1 * x  # E(Y) Simple Linear Regression Function
  ep = y - ey  # Y - E(Y) Regression Error
  ar = np.abs(ep)  # Absolute Residuals
  se = ep ** 2  # Squared Error
  sar = sum(ar)  # Sum of the Absolute Residuals
  sse = sum(se)  # Sum of the Squared Errors
  if lx <= 10:
    epH = "   " + epsilon
    arH = " |" + epsilon + "|"
    seH = "  " + epsilon + squared
    dat = {'   X':x, '   Y':y, 'E(Y)':ey, epH:ep, arH:ar, seH:se}
    df = pd.DataFrame(dat)
    print(f"   Expected Value E(Y)")
    print(f"   Regression Error {epsilon}")
    print(f"   Squared Error {epsilon}{squared}")
    print(df)

  print(f"Sum of Absolute Residuals {Sigma}|{epsilon}| = {sar:7,}")
  print(f"Sum of Squared Errors {Sigma}{epsilon}{squared} = {sse:7,}")
        
#==================================================================================================

def Section5():
  section = "Python-Function 5.1.1: linregress(x,y)"

  if section == "":
    print(f"{0:.3f}")

  elif section == "":
    print(f"{0:.3f}")

  elif section == "Python-Function 5.1.1: linregress(x,y)":
    x = np.array([0, 3, 7, 10])
    y = np.array([5, 5, 27, 31])
    dat = st.linregress(x,y)
    df = pd.DataFrame(FormatLinregress(dat))
    print(df)

  elif section == "SLR PA5.1.9: Calculating sum of squared errors for a regression line":
    x, y = [0, 3, 7, 10], [5, 5, 27, 31]
    b0, b1 = 2, 3

  elif section == "SLR Example 5.1.1: Computing the sum of squared errors":
    x, y = [[0, 3, 7, 10], [5, 5, 27, 31]]
    b0, b1 = 7, 2

  elif section == "SLR PA5.1.6: Making predictions":
    x, y = [2000, 2200], [271000, 275000]
    b0, b1 = 190000, 40

  if section[:3] == "SLR":
    x, y = np.array(x), np.array(y)
    SimpleLinearRegression(x, y, b0, b1)



if(__name__ == "__main__"):
  print("Imports loaded!")
  Section5()
  print("Goodbye!")