from Basic import *
from Module5LinearRegression import SimpleLinearRegression

def PlotResiduals(xs:pd.DataFrame, y:pd.DataFrame, b:np.ndarray=None) -> None:
  """
  """
  pass

def MultipleRegressionAssumptions(xs:pd.DataFrame, y:pd.DataFrame, b:np.ndarray=None) -> None:
  """
  """
  pass

def CorrelationMatrix(xs:pd.DataFrame, y:pd.DataFrame) -> None:
  """
  """
  pass

def MultipleLinearRegression(xs:pd.DataFrame, y:pd.DataFrame, b:np.ndarray=None, tail:int=None, a:float=0.05, sig:float=6) -> None:
  """ Section 6.1.1 Introduction to multiple regression
        A multiple regression model has two parts. The first part is the population linear
        regression function, which represents the expected value of Y given a particular set of
        values for X[1:n]. The population linear regression function is E(Y) = B[0] + sum(B[1:n] *
        X[1:n]), where B[0:n] are regression parameters. The second part is the regression error
        term, epsilon, which represents the difference between the actual value of Y and the
        expected value of Y given a particular set of values for X[1:n].
        The multiple regression model is Y = B[0] + sum(B[1:n] * X[1:n]) + epsilon, which is the
        sum of the population regression function and the regression error term.

      Section 6.1.2 Estimating the multiple regression model
        Once a population linear regression model Y = B[0] + sum(B[1:n] * X[1:n]) + epsilon is
        proposed, the next step is to estimate the model using sample data to find a sample
        multiple regression function. The most common method to estimate the model is called
        "least squares." Intuitively, the least squares method finds the values of the regression
        parameters, B[0:n] that minimize the sum of squared regression errors. The sample multiple
        regression function is Yhat = b[0] + sum(b[1:n] * X[1:n]), where Yhat is the fitted response
        value and b[1:n] are the estimates for B[0:n] that minimize the sum of squared errors. The
        "hat" notation in Yhat is a statistical convention that denotes a sample estimate.


  """
  #SimpleLinearRegression(xs[0], y, b[0], b[1], tail, a, sig)
  n = len(y)
  for i, x in enumerate(xs.count()):
    if n != x:
      raise ValueError(f"X['{xs.columns[i]}'] with {x} entries is {
      'smaller' if x < n else 'larger'} than Y with {n} entries!")
  if b is None:
    xs.insert(0, 'Intercept', 1)
    xT = xs.T
    gram : np.ndarray = np.matmul(xT, xs)
    moment : np.ndarray = np.matmul(xT, y)
    grami : np.ndarray = np.linalg.inv(gram)
    b = np.matmul(grami, moment)
    xs = xs[xs.columns[1:]]

  print(b)
  print("All clear!")

  print(xs)

#==================================================================================================

def Section6():
  section = "MLR test"

  if section == "":
    print(f"{0:.3f}")

  elif section == "":
    print(f"{0:.3f}")
    
  elif section == "":
    print(f"{0:.3f}")
    
  elif section == "":
    print(f"{0:.3f}")
    
  elif section == "":
    print(f"{0:.3f}")
    
  elif section == "":
    print(f"{0:.3f}")
    
  elif section == "":
    print(f"{0:.3f}")
    
  elif section == "":
    print(f"{0:.3f}")
    
  elif section == "":
    print(f"{0:.3f}")
    
  elif section == "":
    print(f"{0:.3f}")
    
  elif section == "":
    print(f"{0:.3f}")
    
  elif section == "MLR test":
    fat = pd.read_csv('fat.csv')
    fat.columns = [c[:6] for c in fat.columns]
    y = fat[fat.columns[1]]
    xs = fat[fat.columns[2:]] 
    MultipleLinearRegression(xs, y, None)
    
  elif section == "-sqrt vs -ln plot":
    global plt
    x = np.linspace(0.001, 256, 256000)
    y1 = np.sqrt(x)
    y2 = np.log(x)
    plt.plot(x, y1, color='r')
    plt.plot(x, y2, color='b')
    plt.show()
    
  elif section == "dis(Pass) vs dis(...) test":
    from dis import dis
    def fnPass() -> None:
      pass
    def fnEllipses() -> None:
      ...
    dis(fnPass)
    print("========================")
    dis(fnEllipses)
    
  elif section == "Python-Practice 6.1.2: Multiple regression using Python":
    import statsmodels.formula.api as sms
    fat = pd.read_csv('fat.csv')
    # Response variable
    Y = fat['body_fat_percent']
    # Generates the linear regression model
    # Multiple predictor variables are joined with +
    model = sms.ols('Y ~ triceps_skinfold_thickness_mm + midarm_circumference_cm + thigh_circumference_cm', data = fat).fit()
    # Prints the summary
    print(model.summary())
    # Prints a list of the fitted values for each sample
    print(model.fittedvalues)
    # Prints a list of the residuals for each sample
    print(model.resid)
    
  elif section == "Python-Practice 6.1.1: 3D scatterplot":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    df = pd.read_csv("http://data-analytics.zybooks.com/Cars.csv")
    fig = plt.figure()
    #mplot3d is needed for projection='3d'
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Speed'], df['Quality'], df['Angle'], c='b', marker='o')
    ax.set_xlabel('Speed, X1')
    ax.set_ylabel('Angle, X2')
    ax.set_zlabel('Quality, Y')
    plt.show()
    

if(__name__ == "__main__"): 
  print("Imports loaded!\n")
  Section6()
  print("\nGoodbye!")