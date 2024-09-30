from Basic import *


def FormatLinregress(inp):
  slope, intercept, r, p, se = inp
  istderr = inp.intercept_stderr
  names = ['Slope', 'Intercept', 'R Value', 'P Value', 'Standard Error', 'Intercept Standard Error']
  values = [slope, intercept, r, p, se, istderr]
  return {'Names':names, 'Values':values}

def SimpleLinearRegression(x:np.ndarray, y:np.ndarray, b0:float=None, b1:float=None, tail:int=0, a:float=0.05, sig:float=6):
  """ Section 5.1.1 Regression Lines
        A simple linear regression is a way to model the linear relationship between two
        quantitative variables, using a line drawn through those variables' data points,
        known as a regression line.

      Section 5.1.5 Least squares method
        Summing the absolute errors is one method to measure how far the line is from the points.
        Another method to measure error is by computing the sum of squared errors. The sum of
        squared errors is the sum of the differences between the Y values of the data points and
        the values obtained from the regression line.

      Section 5.3.3 Coefficient of determination
        Another quantity that shows how well a regression equation represents the data is the
        coefficient of determination. The coefficient of determination, denoted by R^2, gives the
        ratio of the variance in the response variable explained by the predictor variable.
        Conceptually, the coefficient of determination is a measure of how closely the regression
        line follows the pattern of the data. The farther the actual data points are from the
        regression line, the less useful the line actually is in predicting the value of the
        response variable.
      
      Section 5.4.2 Interpreting residual standard error
        A number of quantities use the residuals, epsilon_i = Y_i - Yhat_i:
          * The residual sum of squares is the sum of squared residuals for the sample,
          SSE = sum(Y_i - Yhat_i)^2. The residual sum of squares is typically denoted SSE because
          the residuals are estimated errors. The notation SSR is used for regression sum of
          squares, which is defined in a later section.
          * The residual degrees of freedom is n - p, where p is the number of regression
          parameters. Ex: Simple linear regression has p = 2 regression parameters, so the residual
          degrees of freedom is n - 2. Residual degrees of freedom is often called error degrees of
          freedom.
          * The residual mean square is the residual sum of squares divided by the residual degrees
          of freedom, MSE = SSE / (n - p). Ex: For simple linear regression, MSE = SSE / (n - 2).
          * The residual standard error is the square root of the residual mean square, s = sqrt(MSE).
          The residual standard error, s, estimates the standard deviation of the residuals. The
          measurement unit of the residual standard error is the same as the measurement unit of
          the response Y variable.
        The residual sum of squares, residual degrees of freedom, and residual mean square are used
        in an Analysis of Variance table, which is often abbreviated as "ANOVA table." A later
        section shows how an ANOVA table is used in model assessment.
        The residual standard error is a measure of the precision of a model prediction. The sample
        simple linear regression line can be used to predict a future value of Y for a fixed value
        of X. A relatively small residual standard error indicates that the actual future value of
        Y is likely to be relatively close to the predicted value. Therefore, less residual
        standard error is better.

      Section 5.5.6 The coefficient of determination
        The coefficient of determination, denoted by R^2 is another measure of correlation. The
        coefficient of determination is useful because the quantity measures the proportion of
        total variation in the response variable, Y, that is accounted for by the linear regression
        model. Intuitively, the value of R^2 can be viewed as a quantitative way of measuring
        certainty when making predictions from a model. Although R^2 can easily be calculated by
        squaring the Pearson correlation coefficient, R^2 can also be calculated from the values in
        the ANOVA table: R^2 = (SSTO - SSE) / SSTO = SSR / SSTO.
        SSTO and SSE measure different quantities:
        * The best prediction of Y if one ignores X is the sample mean of Y, Ybar. Thus, the total
        sum of squares, SSTO = Sigma(Y_i - Ybar)^2 measures the variation in Y ignoring X.
        * The best prediction of Y based on the simple linear regression model is Yhat. Thus, the
        residual sum of squares, SSE = Sigma(Y_i - Yhat)^2, measures the variation remaining in Y
        after using X to predict Y.
        The value of R^2 is typically expressed as a percentage between 0% and 100%:
        * If a strong linear relationship exists between Y and X, using X to predict Y will leave
        little variation remaining in Y. Then SSE will be small and R^2 will be high (generally
        greater than 90%).
        * Conversely, if a linear relationship does not exist between Y and X, using X to predict
        Y will leave a lot of variation remaining in Y. Then SSE will be close to SSTO and R^2 will
        be low (close to 0%).
  """
  n, ly = len(x), len(y)
  if n != ly:
    print("X Y mismatched!")
    return None
  ybar = y.mean()     # Mean of the data's response variable
  xbar = x.mean()     # Mean of the data's predictor variable
  dx = x - xbar       # Deviation of x_i from the mean of x
  dy = y - ybar       # Deviation of y_i from the mean of y
  if b0 is None and b1 is None:
    b1 = np.sum(dx * dy) / np.sum(dx ** 2)
    b0 = ybar - b1 * xbar
  yhat = b0 + b1 * x  # E(Y) for pop, Yhat for sample, Simple Linear Regression Function
  ep = y - yhat       # Y - E(Y), Regression Error
  ar = np.abs(ep)     # Absolute Residual
  se = ep ** 2        # Squared Error
  if n <= 10:
    yhH = f"  Y{hat}"
    print(f"   Expected Value E(Y) = Y{hat}")
    epH = f"   {epsilon}"
    print(f"   Regression Error {epsilon} = Y - Y{hat}")
    arH = f"|{epsilon}|"
    print(f"   Absolute Regression Error |{epsilon}|")
    seH = f" {epsilon}{squared}"
    print(f"   Squared Error {epsilon}{squared}")
    dat = pd.DataFrame({'  X':x, '  Y':y, yhH:yhat, epH:ep, arH:ar, seH:se})
    print(dat)

  # Creating ANOVA Table
  p = 2                             # The number of parameters, for SLR this is 2 (beta_0 and beta_1)
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
  print(f"Y-Intercept of Regression Line  {beta}{sub0} = {b0:{dmt}}")
  print(f"Slope of Regression Line        {beta}{sub1} = {b1:{dmt}}")
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

  r = CorrelationCoefficient(x, y, n)
  print(f"Sqrt(R{squared}) = {r2 ** 0.5:.{sig}f}")
  print(f"     R   = {r:.{sig}f}")
  PopCorrTTest(r, df + 1, 0, 0.10, sig)

def CorrelationCoefficient(x:np.ndarray, y:np.ndarray, verbose:int = 0) -> float:
  """ Section 5.3.1 Correlation and coefficient of determination
        Correlation describes the association or dependence between two variables. A positive
        correlation between two variables means that as one variable increases, the other variable
        increases as well. A negative correlation between two variables means that as one variable
        increases, the other variable decreases. The strength of correlation between a predictor
        variable and a response variable can be measured by the correlation coefficient. The
        population correlation coefficient is denoted by rho and the sample correlation coefficient
        is denoted by R. The strength of correlation can be described by the absolute value of R.
  """
  n, ly = len(x), len(y)
  if n != ly:
    print("X Y mismatched!")
    return None
  xy, x2, y2 = x * y, x ** 2, y ** 2
  v = [x, y, xy, x2, y2]
  s = [np.sum(vi) for vi in v]
  sx, sy, sxy, sx2, sy2 = s
  rn = n * sxy - sx * sy
  rdx = (n * sx2 - sx ** 2) ** 0.5
  rdy = (n * sy2 - sy ** 2) ** 0.5
  r = rn / (rdx * rdy)
  if verbose > 0:
    h = ['X', 'Y', 'XY', 'X2', 'Y2']
    dat = dict(zip(h, v))
    # dat = {'X':x, 'Y':y, 'XY':xy...}
    sums = dict(zip(h, s))
    df = pd.DataFrame(dat)
    df.loc[Sigma] = pd.Series(sums)
    print(df.head(verbose))
    print(f"{r =}")
    ar = abs(r)
    print("No" if ar < 0.001 else ((
          "Perfect" if ar > 0.999 else
          "Strong" if ar > 0.80 else
          "Moderate" if ar > 0.40 else
          "Weak") + (" Positive" if r > 0 else
          " Negative")), "Correlation")
  return r

def PopCorrTTest(r:float, n:int, tail:int, a:float, sig:int=3):
  """ Section 5.3.2 t-test for the population correlation coefficient
      The distribution of the Pearson correlation coefficients for samples of size n follows a
      t-distribution with n - 2 degrees of freedom. When determining whether a linear relationship
      or association exist, the t-test for population correlation coefficient is useful.
      The t-test for the population correlation coefficient is performed as follows.
  """
  df = n - 2
  t = r * df ** 0.5 / (1 - r ** 2) ** 0.5
  TTest(t, df, tail, a, sig)

def ConfidenceInterval(c:float, sampleMean:float, std:float, n:int, pop:bool, p:int = 1):
  """ Section 5.5.4: Confidence intervals for regression parameters
      The slope estimator, b_1, provides a single number to estimate beta_1. To quantify sampling
      uncertainty about a single number used to estimate beta_1, one can calculate an interval
      around the number. A confidence interval for the slope is an interval around b_1 that
      quantifies sampling uncertainty when b_1 is used to estimate beta_1. The confidence interval
      is given by [b_1 - t*(SE), b_1 + t*(SE)] where SE is the standard error, and t* depends on
      the degrees of freedom and the confidence level of interest, and can be found from a
      t-distribution table. Although far less common, the confidence interval for the intercept
      can also be calculated.
  """
  from Module3ConfidenceIntervals import Tstar
  sampleStd = std
  print(f"{sampleStd = }")
  criticalValue = Tstar(c, n - p)
  print(f"{criticalValue = }")
  standardError = std# / (n ** 0.5)
  print(f"{standardError = }")
  m = criticalValue * standardError
  print(f"{m = }")
  return (sampleMean - m, sampleMean + m)





#==================================================================================================

def Section5():
  section = "Module 5 Discussion"

  if section == "":
    from Calculus import CalcTest
    CalcTest()

  elif section == "":
    print(f"{0:.3f}")

  elif section == "Module 5 Discussion":
    csv = ("C:\\Users\\Nihil\\OneDrive - SNHU\\24EZ1 MAT-243 Applied Statistics 1 for STEM"
           "\\5-3 Discussion Simple Linear Regression\\Module 5 Discussion.csv")
    df = pd.read_csv(csv)
    x, y = df['wt'], df['mpg']
    SimpleLinearRegression(x, y)
    import matplotlib.pyplot as plt
    plt.plot(x, y, 'o', color='red')
    plt.title('MPG against Weight')
    plt.xlabel('Weight (1000s lbs)')
    plt.ylabel('MPG')
    plt.show()

  elif section == "PA5.5.9: Finding and Interpreting the coefficient of determination":
    df = pd.read_csv("http://data-analytics.zybooks.com/gpa.csv")
    x, y = df['height'], df['gpa']
    SimpleLinearRegression(x, y)

  elif section == "Python-Practice 5.5.1: Using ANOVA to test the correlation between two variables":
    from statsmodels.formula.api import ols
    import statsmodels.api as sm
    scores = pd.read_csv("ExamScores.csv")
    x, y = scores['Exam1'], scores['Exam4']
    SimpleLinearRegression(x, y)
    mod = ols('Exam4 ~ Exam1',scores).fit()
    print(sm.stats.anova_lm(mod, typ=2))

  elif section == "Example 5.5.2: Finding a confidence interval for the slope":
    c = 0.99
    n = 50
    p = 2
    sampleMean = 0.1788
    std = 0.077
    ci = ConfidenceInterval(c, sampleMean, std, n, False, p)
    print(f"{FormatCI(ci)}")
    print()
    scores = pd.read_csv("ExamScores.csv")
    x, y = scores['Exam1'], scores['Exam4']
    #b0, b1 = 62.3017, 0.1788
    b0, b1 = 57.7627, 0.2266
    SimpleLinearRegression(x, y, b0, b1)

  elif section == "PA5.5.2: Testing a simple linear regression slope":
    b1, seb1 = 0.1788, 0.077
    t = b1 / seb1
    df = 12 - 1
    TTest(t, df, 0, 0.05, 3)

  elif section == "Python-Function 5.5.1: ols(), fit(), and summary()":
    import statsmodels.formula.api as smf
    scores = pd.read_csv("ExamScores.csv")
    model = smf.ols('Exam4 ~ Exam2', scores).fit()
    print(model.summary())
    b1, seb1 = 0.1788, 0.077
    t = b1 / seb1
    df = len(scores['Exam4']) - 1
    TTest(t, df, 0, 0.05, 3)

  elif section == "PA5.4.3: Interpreting the residual standard error":
    print(f"{beta}{squared}{sub0}{epsilon}{hat}{subi}{theta}{bar} 12{frac}34")
    print(f"{Sigma}Y{bar} = Y{hat}{subi}{squared}")

    x = [ 0, 1,  3,  6,  6,  8]
    y = [6, 12, 18, 33, 42, 57]
    b0, b1 = 4, 6
    x, y = np.array(x), np.array(y)
    SimpleLinearRegression(x, y, b0, b1)

  elif section == "Timeit":
    from timeit import timeit
    ft = timeit('print(f"{nums[0]}, {nums[1]}")',#, {nums[2]}, {nums[3]}, {nums[4]}, '
           #'{nums[5]}, {nums[6]}, {nums[7]}, {nums[8]}, {nums[9]}")', 
           'nums = [x for x in range(10)]')
    pt = timeit('print(nums[0], ",", nums[1])',#, ",", nums[2], ",", nums[3], ",", nums[4],'
           #'",", nums[5], ",", nums[6], ",", nums[7], ",", nums[8], ",", nums[9])', 
           'nums = [x for x in range(10)]')
    print()
    print(ft)
    print(pt)

  elif section == "Python-Practice 5.3.1: Coefficient of determination":
    from statsmodels.formula.api import ols
    import statsmodels.api as sm
    # The ExamScores dataset is loaded
    scores = pd.read_csv("ExamScores.csv")
    # Creates a linear regression model
    results = ols('Exam4 ~ Exam1', data=scores).fit()
    # Prints the results
    print(results.summary())
    # Creates an analysis of variance table
    aov_table = sm.stats.anova_lm(results, typ=2)
    # Prints the analysis of variance table
    print(aov_table)
    x = scores['Exam1']
    y = scores['Exam4']
    b0, b1 = 57.7627, 0.2266
    SimpleLinearRegression(x, y, b0, b1)

  elif section == "Python-Function 5.3.1: scipy.stats.pearsonr(x,y)":
    scores = pd.read_csv("ExamScores.csv")
    x, y = scores['Exam1'], scores['Exam4']
    print(st.pearsonr(x, y))

  elif section == "Example 5.3.2: Using the t-test for population correlation coefficient":
    scores = pd.read_csv("ExamScores.csv")
    x, y = scores['Exam1'], scores['Exam4']
    r = CorrelationCoefficient(x, y, 5)
    print(f"{r = }")
    n, tail, a, sig = 50, 1, 0.05, 3
    PopCorrTTest(r, n, tail, a, sig)

  elif section == "zyDE 5.3.1: corr()":
    scores = pd.read_csv("ExamScores.csv")
    print(scores[['Exam1','Exam2']].corr())
    print(scores[['Exam1','Exam2','Exam3','Exam4']].corr())

  elif section == "Example 5.3.1: Calculating the correlation coefficient using the formula":
    x = np.array([1, 2, 4, 5])
    y = np.array([1, 3, 5, 7])
    CorrelationCoefficient(x, y, True)

  elif section == "Python-Function 5.1.1: linregress(x,y)":
    x = np.array([0, 3, 7, 10])
    y = np.array([5, 5, 27, 31])
    dat = st.linregress(x,y)
    df = pd.DataFrame(FormatLinregress(dat))
    print(df)

  elif section == "SLR PA5.1.9: Calculating sum of squared errors for a regression line":
    x, y = [0, 3, 7, 10], [5, 5, 27, 31]
    b0, b1 = 2, 3
    x, y = np.array(x), np.array(y)
    SimpleLinearRegression(x, y, b0, b1)

  elif section == "SLR Example 5.1.1: Computing the sum of squared errors":
    x, y = [[0, 3, 7, 10], [5, 5, 27, 31]]
    b0, b1 = 7, 2
    x, y = np.array(x), np.array(y)
    SimpleLinearRegression(x, y, b0, b1)

  elif section == "SLR PA5.1.6: Making predictions":
    x, y = [2000, 2200], [271000, 275000]
    b0, b1 = 190000, 40
    x, y = np.array(x), np.array(y)
    SimpleLinearRegression(x, y, b0, b1)



if(__name__ == "__main__"):
  print("Imports loaded!\n")
  Section5()
  print("\nGoodbye!")