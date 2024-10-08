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
  lx = len(xs.columns)
  for i, x in enumerate(xs.count()):
    if n != x:
      raise ValueError(f"X['{xs.columns[i]}'] with {x} entries is {
      'smaller' if x < n else 'larger'} than Y with {n} entries!")
  if b is None:
    xs.insert(0, '1', 1)
    xT = xs.T
    gram : np.ndarray = np.matmul(xT, xs)
    moment : np.ndarray = np.matmul(xT, y)
    grami : np.ndarray = np.linalg.inv(gram)
    b = np.matmul(grami, moment)
    #xs = xs[xs.columns[1:]]
    #print(b)
  #print(xs)
  # E(Y) for pop, Yhat for sample, Multiple Linear Regression Function
  yhat = sum([b[i] * xs[c] for i, c in enumerate(xs.columns)])

  # COPIED
  ybar = y.mean()     # Mean of the data's response variable
  dy = y - ybar       # Deviation of y_i from the mean of y
  ep = y - yhat       # Y - E(Y), Regression Error
  ar = np.abs(ep)     # Absolute Residual
  se = ep ** 2        # Squared Error

  yhH = f"  Y{hat}"
  epH = f"   {epsilon}"
  arH = f"|{epsilon}|"
  seH = f" {epsilon}{squared}"
  dat = xs.join(pd.DataFrame({'  Y':y, yhH:yhat, epH:ep, arH:ar, seH:se}))
  if n <= 10:
    print(f"   Expected Value E(Y) = Y{hat}")
    print(f"   Regression Error {epsilon} = Y - Y{hat}")
    print(f"   Absolute Regression Error |{epsilon}|")
    print(f"   Squared Error {epsilon}{squared}")
    print(dat)

  # Creating ANOVA Table
  p = len(b)                        # The number of parameters, for SLR this is 2 (beta_0 and beta_1)
  sar = np.sum(ar)                  # Sum of the Absolute Residuals
  sse = np.sum(se)                  # Residual Sum of Squares (Sum of the Squared Errors, Unxplained Variance)
  resdf = n - p                     # Residial Degrees of Freedom
  mse = sse / resdf                 # Residual Mean Square (Mean Square Error)
  s = mse ** 0.5                    # Residual Standard Error
  epbar = ep.mean()                 # Mean of Regression Errors
  ssr = np.sum((yhat - ybar) ** 2)  # Regression Sum of Squares (Explained Variance)
  tvar = np.sum((y - ybar) ** 2)    # Total Variance
  regdf = p - 1                     # Regression Degrees of Freedom
  msr = ssr / regdf                 # Regression Mean Square, for SLR (MSR=SSR)
  ssto = ssr + sse                  # Total Sum of Squares
  df = regdf + resdf                # Total Degrees of Freedom = n - 1
  r2 = ssr / tvar                   # Coefficient of Determination R^2
  f = msr / mse                     # ANOVA F-statistic
  maxlen = int(np.log10(ssto) + 1)  # To format, take the length of the longest value
  intlen = maxlen + maxlen // 3     # Make room for commas
  declen = int(intlen + 1 + sig)    # Make room for decimals
  fmt = f"{intlen},"                # Format the integers into a string
  dmt = f"{declen},.{sig}f"         # Format the decimals into a string
  print(f"ANOVA Table:                       = ")
  print(f"Y-Intercept of Regression Line  {beta}{subn[0]} = {b[0]:{dmt}}")
  for i in range(1, lx + 1):
    print(f"Slope of Regression Line {i}      {beta}{subn[i]} = {b[i]:{dmt}}")
  print(f"Number of Samples                n = {n:{fmt}}")
  print(f"Parameters                       p = {p:{fmt}}")
  print(f"Sum of Absolute Residuals     {Sigma}|{epsilon}| = {sar:{dmt}}")

  print(f"Residual Sum of Squares        {Sigma}{epsilon}{squared} = {sse:{dmt}}")
  print(f"Residual Degrees of Freedom    n-p = {resdf:{fmt}}")
  print(f"Residual Mean Square           MSE = {mse:{dmt}}")
  print(f"Regression Sum of Squares      SSR = {ssr:{dmt}}")
  print(f"Regression Degrees of Freedom  p-1 = {regdf:{fmt}}")
  print(f"Regression Mean Square         MSR = {msr:{dmt}}")
  print(f"Total Degrees of Freedom       n-1 = {df:{fmt}}")
  print(f"ANOVA F-statistic                F = {f:{dmt}}")

  print(f"Residual Standard Error          s = {s:{dmt}}")
  print(f"Mean of Regression Errors        {epsilon}{bar}= {epbar:{dmt}}")
  print(f"Mean of Sample Values            Y{bar}= {ybar:{dmt}}")
  print(f"Explained Variance       {Sigma}(Y{hat}-Y{bar}){squared} = {ssr:{dmt}}")
  print(f"Unexplained Variance      {Sigma}(Y-Y{hat}){squared} = {sse:{dmt}}")
  print(f"Total Variance            {Sigma}(Y-Y{bar}){squared} = {tvar:{dmt}}")
  print(f"Total Sum of Squares  SSR+SSE=SSTO = {ssto:{dmt}}")
  print(f"Coefficient of Determination    R{squared} = {r2:{dmt}}")
  print(f"{r2*100: .1f}% of variation in Y is accounted for by X")



#==================================================================================================

def Section6():
  #global plt
  section = "Python-Practice 6.2.1: Residual plots"

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
    
  elif section == "Python-Practice 6.2.1: Residual plots":
    import statsmodels.formula.api as sms

    fat = pd.read_csv('fat.csv')

    # Response variable
    Y = fat['body_fat_percent']

    # Generates the linear regression model
    # Multiple predictor variables are joined with +
    model = sms.ols('Y ~ triceps_skinfold_thickness_mm + midarm_circumference_cm + thigh_circumference_cm', data = fat).fit()

    plt.figure(figsize = (20, 16))
    plt.tight_layout()

    plt.subplot(2, 2, 1)
    plt.scatter(x = fat['triceps_skinfold_thickness_mm'], y = model.resid, color = 'blue', edgecolor = 'k')
    xmin = min(fat['triceps_skinfold_thickness_mm'])
    xmax = max(fat['triceps_skinfold_thickness_mm'])
    plt.hlines(y = 0, xmin = xmin, xmax = xmax, color = 'red', linestyle = '--')
    plt.xlabel('$X_1$', fontsize = 16)
    plt.ylabel('Residuals', fontsize = 16)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.title('$X_1$ vs. residuals', fontsize = 24)

    plt.subplot(2, 2, 2)
    plt.scatter(x = fat['midarm_circumference_cm'], y = model.resid, color = 'blue', edgecolor = 'k')
    xmin = min(fat['midarm_circumference_cm'])
    xmax = max(fat['midarm_circumference_cm'])
    plt.hlines(y = 0, xmin = xmin, xmax = xmax, color = 'red', linestyle = '--')
    plt.xlabel('$X_2$', fontsize = 16)
    plt.ylabel('Residuals', fontsize = 16)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.title('$X_2$ vs. residuals', fontsize = 24)

    plt.subplot(2, 2, 3)
    plt.scatter(x = fat['thigh_circumference_cm'], y = model.resid, color = 'blue', edgecolor = 'k')
    xmin = min(fat['thigh_circumference_cm'])
    xmax = max(fat['thigh_circumference_cm'])
    plt.hlines(y = 0, xmin = xmin, xmax = xmax, color = 'red', linestyle = '--')
    plt.xlabel('$X_3$', fontsize = 16)
    plt.ylabel('Residuals', fontsize = 16)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.title('$X_3$ vs. residuals', fontsize = 24)

    plt.subplot(2, 2, 4)
    plt.scatter(x = model.fittedvalues, y = model.resid, color = 'blue', edgecolor = 'k')
    xmin = min(Y)
    xmax = max(Y)
    plt.hlines(y = 0, xmin = xmin, xmax = xmax, color = 'red', linestyle = '--')
    plt.xlabel('Fitted values', fontsize = 16)
    plt.ylabel('Residuals', fontsize = 16)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.title('Fitted values vs. residuals', fontsize = 24)
    plt.show()

    
  elif section == "Testing SLR vs MLR":
    fat = pd.read_csv('fat.csv')
    fat.columns = [c[:6] for c in fat.columns]
    y = fat[fat.columns[1]]
    xs = fat[fat.columns[2]] 
    SimpleLinearRegression(xs, y)
    xs = fat[fat.columns[2:3]] 
    MultipleLinearRegression(xs, y, None)

    
  elif section == "MLR test":
    fat = pd.read_csv('fat.csv')
    fat.columns = [c[:6] for c in fat.columns]
    y = fat[fat.columns[1]]
    xs = fat[fat.columns[2:5]] 
    MultipleLinearRegression(xs, y, None)
    
  elif section == "-sqrt vs -ln plot":
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