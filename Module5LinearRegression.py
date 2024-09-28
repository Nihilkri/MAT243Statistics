from Basic import *



def SimpleLinearRegression(dat:list, b0:float, b1:float):
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
  xs, ys = dat
  lx, ly = len(xs), len(ys)
  if lx != ly:
    print("X Y mismatched!")
    return None
  sse = 0
  for x, y in zip(xs, ys):
    e = b0 + b1 * x
    ep = y - e
    se = ep ** 2
    sse += se
    print(f"{f"Expected Value E({x:4}) = ":28}{e:7,}")
    print(f"{f"Regression Error {epsilon} = ":28}{ep:7,}")
    print(f"{f"Squared Error {epsilon}^2 = ":28}{se:7,}")
  print(f"{f"Sum of Squared Errors {Sigma}{epsilon}^2 = ":28}{sse:7,}")
        
#==================================================================================================

def Section5():
  section = "PA5.1.9: Calculating sum of squared errors for a regression line"

  if section == "":
    print(f"{0:.3f}")

  elif section == "":
    print(f"{0:.3f}")

  elif section == "":
    print(f"{0:.3f}")

  elif section == "PA5.1.9: Calculating sum of squared errors for a regression line":
    dat = [[0, 3, 7, 10], [5, 5, 27, 31]]
    SimpleLinearRegression(dat, 2, 3)

  elif section == "Example 5.1.1: Computing the sum of squared errors":
    dat = [[0, 3, 7, 10], [5, 5, 27, 31]]
    SimpleLinearRegression(dat, 7, 2)

  elif section == "PA5.1.6: Making predictions":
    dat = [[2000, 2200], [271000, 275000]]
    SimpleLinearRegression([(2000, 271000), (2200, 275000)], 190000, 40)



if(__name__ == "__main__"):
  print("Imports loaded!")
  Section5()
  print("Goodbye!")