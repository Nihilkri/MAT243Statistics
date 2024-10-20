from Basic import *
from Module5LinearRegression import SimpleLinearRegression

def PlotResiduals(xs:pd.DataFrame, y:pd.DataFrame, b:np.ndarray=None) -> None:
  """
  """
  pass

def MultipleRegressionAssumptions(xs:pd.DataFrame, y:pd.DataFrame, b:np.ndarray=None) -> None:
  """ Section 6.2.1 Assumptions of the multiple regression model
        A multiple regression model is considered valid only if the following assumptions can be
        made about the population. Since population regression errors are not observable, the
        sample residuals e_i = Y_i - Yhat_i are used to determine whether each assumption is
        violated.
        Mean of zero: The mean of each residual for each set of values for the predictor variables
        is zero. Equivalently, this assumption says that the response variable is a linear function
        of each of the predictor variables.
        Independence: The residuals are independent. This condition can be difficult to assess. A
        common way to determine independence is by plotting residuals with respect to the time in
        which the data is collected. If a trend exists, then the independence assumption is
        potentially violated.
        Normality: The residuals of each set of values for the predictor variables form a normal
        distribution. If the plotted points lie reasonably close to the diagonal line on the plot
        then one can conclude that the normality assumption holds.
        Constant variance: The residuals of each set of values for the predictor variables should
        have equal or similar variance. A common term for this condition is homoscedasticity. If
        the variance does not remain constant throughout the plot, then the model exhibits
        heteroscedasticity.
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

        Section 7.1 Interpreting multiple regression models
          
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
  epbar = ep.mean()                 # Mean of Regression Errors

  sse = np.sum(se)                  # Residual Sum of Squares (Sum of the Squared Errors, Unxplained Variance)
  resdf = n - p                     # Residial Degrees of Freedom
  mse = sse / resdf                 # Residual Mean Square (Mean Square Error)
  ssr = np.sum((yhat - ybar) ** 2)  # Regression Sum of Squares (Explained Variance)
  regdf = p - 1                     # Regression Degrees of Freedom
  msr = ssr / regdf                 # Regression Mean Square, for SLR (MSR=SSR)
  df = regdf + resdf                # Total Degrees of Freedom = n - 1
  f = msr / mse                     # ANOVA F-statistic

  rmse = mse ** 0.5                 # Residual Mean Square Error (Residual Standard Error)
  ssto = np.sum((y - ybar) ** 2)    # Total Sum of Squares = Total Variance
  tvar = ssr + sse                  # Total Variance = Total Sum of Squares
  r2 = ssr / ssto                   # Coefficient of Determination R^2
  r2adj = 1 - (1 - r2) * (n - 1) / (n - (lx + 1))
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
  print(f"Mean of Sample Values            Y{bar}= {ybar:{dmt}}")
  print(f"Mean of Regression Errors        {epsilon}{bar}= {epbar:{dmt}}")
  print(f"Sum of Absolute Residuals     {Sigma}|{epsilon}| = {sar:{dmt}}")
  print()
  print(f"Residual Sum of Squares  {Sigma}{epsilon}{squared} = SSE = {sse:{dmt}}")
  print(f"Residual Degrees of Freedom    n-p = {resdf:{fmt}}")
  print(f"Residual Mean Square           MSE = {mse:{dmt}}")
  print(f"Regression Sum of Squares      SSR = {ssr:{dmt}}")
  print(f"Regression Degrees of Freedom  p-1 = {regdf:{fmt}}")
  print(f"Regression Mean Square         MSR = {msr:{dmt}}")
  print(f"Total Degrees of Freedom       n-1 = {df:{fmt}}")
  print(f"ANOVA F-statistic                F = {f:{dmt}}")
  print()
  print(f"Residual Standard Error       RMSE = {rmse:{dmt}}")
  print(f"Explained Variance       {Sigma}(Y{hat}-Y{bar}){squared} = {ssr:{dmt}}")
  print(f"Unexplained Variance      {Sigma}(Y-Y{hat}){squared} = {sse:{dmt}}")
  print(f"Total Variance            {Sigma}(Y-Y{bar}){squared} = {tvar:{dmt}}")
  print(f"Total Sum of Squares  SSR+SSE=SSTO = {ssto:{dmt}}")
  print(f"Coefficient of Determination    R{squared} = {r2:{dmt}}")
  print(f"{r2*100: .1f}% of variation in Y is accounted for by X")
  print(f"Adj Coef. of Determination   R{squared}adj = {r2adj:{dmt}}")



#==================================================================================================

def Section6():
  #global plt
  section = "Alexus's Lab 7"

  if section == "":
    print(f"{0:.3f}")

    if section == "":
      print(f"{0:.3f}")
    
    elif section == "":
      print(f"{0:.3f}")
    
    elif section == "":
      print(f"{0:.3f}")
    
    elif section == "":
      print(f"{0:.3f}")
    
  elif section == "Alexus's Lab 7":
    dat = pd.Series([4, 6, 7, 2, 5, 4, 7, 9])
    print(f"Step 1: H_0: {mu} = ~~~~~~~~~~~~~")
    print(f"Step 2: H_a: {mu} {ne} ~~~~~~~~~~~~~")
    from Module3ConfidenceIntervals import Zstar, HypTest
    print(f"Step 3: Z* = {Zstar(0.90)}")
    print(f"Step 4: {mu} = {dat.mean():.3f}")
    print(f"Step 5: {sigma} = {dat.std():.3f}")
    df = pd.DataFrame({}).s

    
  elif section == "PA7.1.2: Interpreting residual standard error":
    print(f"{0:.3f}")
    
  elif section == "Python-Practice 7.1.1: Multiple regression models":
    """ Consider the body fat dataset and a model where the response variable Y is percent body fat
          and the predictor variables X_1 = triceps skinfold thickness (mm) and X_2 = midarm
          circumference (cm). The model is constructed using the code below.

        R-squared measures the proportion of total variation in Y that is accounted for by the
          multiple regression model, which is 0.786. Adj. R-squared is an adjustment to R-squared
          that allows alternative models for the same response variable to be compared. F-statistic
          and Prob (F-statistic) tests whether no linear regression relationship exists between Y
          and the the set {X1, X2}.

        The coef column in the table below are the estimates for the parameters, which are
          b0 = 6.7916, b1 = 1.0006, and b2 = -0.4314. Thus, the equation for the model is
          Yhat = 6.7916 + 1.0006 X_1 - 0.4314 X_2. The std err column contains standard errors of
          the regression parameter estimators, which measure the precision of the estimators. The
          t column contains individual t-statistics for the regression parameter estimators, equal
          to each estimate divided by its standard error. The next column contains individual
          p-values for the regression parameter estimators, equal to the sum of the tail areas
          beyond the t-statistic. The last two columns give the lower and upper bounds of the
          95% confidence interval.
    """
    from statsmodels.formula.api import ols
    fat = pd.read_csv('fat.csv')
    Y = fat['body_fat_percent']
    t1 = 'triceps_skinfold_thickness_mm'
    t2 = 'midarm_circumference_cm'
    X = fat[[t1, t2]]
    m12 = ols(f'Y ~ {t1} + {t2}', data = fat).fit()
    print(m12.summary())
    MultipleLinearRegression(X, Y)
    
  elif section == "Python-Practice 6.2.2: qqplot()":
    import statsmodels.graphics.gofplots as smg
    import statsmodels.formula.api as sms
    fat = pd.read_csv('fat.csv')
    # Response variable
    Y = fat['body_fat_percent']
    # Generates the linear regression model
    # Multiple predictor variables are joined with +
    model = sms.ols('Y ~ triceps_skinfold_thickness_mm + midarm_circumference_cm + thigh_circumference_cm', data = fat).fit()
    fig = smg.qqplot(model.resid, line = '45', fit = 'True')
    plt.xlabel('Theoretical quantiles')
    plt.ylabel('Sample quantiles')
    plt.title('Q-Q plot of normalized residuals')
    plt.show()

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
    mx = len(fat.columns)
    for i in range(1, mx):
      if i == mx - 1:
        name = 'Fitted values'
        df = model.fittedvalues
        xmin = min(Y)
        xmax = max(Y)
      else:
        name = f'$X_{i}$'
        df = fat[fat.columns[i + 1]]
        xmin = min(df)
        xmax = max(df)
      plt.subplot(2, 2, i)
      plt.scatter(x = df, y = model.resid, color = 'blue', edgecolor = 'k')
      plt.hlines(y = 0, xmin = xmin, xmax = xmax, color = 'red', linestyle = '--')
      plt.xlabel(name, fontsize = 16)
      plt.ylabel('Residuals', fontsize = 16)
      plt.xticks(fontsize = 12)
      plt.yticks(fontsize = 12)
      plt.title(f'{name} vs. residuals', fontsize = 24)
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