from operator import index
from Basic import *

def Chi2(dat: pd.DataFrame, proportion: bool) -> None:
  """ Section 8.2.1 Chi-square goodness-of-fit test
        The chi-square distribution can be used to test how close the distribution of a population
        is to a theoretical distribution. The chi-squared test statistic measures how different
        the observed counts are compared to the expected counts, assuming the null hypothesis is
        true. Specifically, Chi**2 = sum((O[i] - E[i]) ** 2 / E[i]).
        
        The difference in observed and expected counts is squared to account for the fact that
        some of the absolute differences will be positive and some will be negative. Squaring the
        differences ensures that all terms will be positive. Dividing by the expected count is
        necessary to scale the differences as a percentage. Ex: If the expected count is 5, a
        squared difference of 10 is 200% of the expected count (10/5 = 2). A squared difference
        of 10 is much less relative to an expected count of 500 (10/500 = 0.02).
        
        The alternate hypothesis for a chi-squared test is always two-sided, meaning that the
        proportions are not equal. Thus, if the null hypothesis is rejected, no conclusion can be
        made on which proportion is larger.
  """
  if proportion:
    t = dat["Observed"].sum()
    dat["Expected"] *= t
  print(dat)
  print()
  k = len(dat.index)
  df = k - 1
  print(f"{df = }")
  lst = []
  for row in dat.index:
    oi = dat["Observed"][row]
    ei = dat["Expected"][row]
    chii = (oi - ei) ** 2 / ei
    print(f"({oi} - {ei}) ** 2 / {ei} = {chii}")
    lst.append(chii)
  chi2 = sum(lst)
  #chi2 = sum([(dat["Observed"][row] - dat["Expected"][row]) ** 2 / dat["Expected"][row] for row in dat.index])
  print(f"{chi2 = :0.3f}")
  pvalue = st.chi2.sf(chi2, df)
  print(f"{pvalue = :0.3f}")

  from scipy.stats import chisquare
  # Calculates the chi-square statistic and $p$-value for alpha = 0.05 for the given observed and expected counts
  statistic, pvalue = chisquare(dat["Observed"], f_exp=dat["Expected"])
  print(f"{statistic = :.4f}")
  print(f"{pvalue = :.4f}")

def ContingencyTable(obs:pd.DataFrame, hasTotals:bool=False) -> None:
  """ Section 8.2.2
        A chi-square test can also be used to determine whether two or more variables are
        independent by comparing the distributions of the variables over two or more categories.
        This test is the chi-square test for independence. The expected counts used to calculate
        the test statistic in this case come from a contingency table. A contingency table is
        constructed from the values of the variables and categories along the rows and columns. An
        expected cell count is calculated by multiplying the row total by the column total and
        dividing by the overall total.
  """
  dct = {}
  if hasTotals:
    tx = obs.columns[-1]
    ty = obs.index[-1]
  else:
    pass
  for x in obs.columns:
    lst = []
    for y in (obs.index[:-1] if hasTotals else obs.index):
      lst.append(obs[tx][y] * (1 if (hasTotals and x == tx) else obs[x][ty] / obs[tx][ty]))
    if hasTotals:
      lst.append(obs[x][ty])
    dct[x] = lst
  exp = pd.DataFrame(dct, index=obs.index)
  print("Observed")
  print(obs)
  print()
  print("Expected")
  print(exp)

def Chi2Test(dat:pd.DataFrame) -> None:


  pass

#==================================================================================================

def Section8():
  section = "Python-Practice 8.2.2: Chi-squared test of independence"

  if section == "":
    print(f"{0:.3f}")

  elif section == "":
    print(f"{0:.3f}")
    
  elif section == "":
    print(f"{0:.3f}")
    
  elif section == "Python-Practice 8.2.2: Chi-squared test of independence":
    from scipy.stats import chi2_contingency
    # Construct a contingency table
    parole = np.array([[405,1422], [240,470], [151,275]])
    # Calculate the test statistic, $p$-value, degrees of freedom, and expected counts
    chi2, p, df, ex = chi2_contingency(parole)
    print(f"{chi2 = }")
    print(f"{p = }")
    print(f"{df = }")
    print(f"{ex = }")
    
  elif section == "PA8.2.2: Calculating expected cell counts":
    from io import StringIO
    csv = StringIO(
      ",Died,Survived,Total by specialty\n"
      "Infantry,374,2629,3003\n"
      "Non-infantry,73,1219,1292\n"
      "Total by outcome,447,3848,4295")
    dat = pd.read_csv(csv, index_col=0)
    ContingencyTable(dat, True)
    
  elif section == "Example 8.2.2: Calculating the expected counts for a 3x2 contingency table":
    from io import StringIO
    csv = StringIO(
      "Punishment,Parole violation,No parole violation,Total by punishment\n"
      "0,405,1422,1827\n"
      "1-2,240,470,710\n"
      "3 or greater,151,275,426\n"
      "Total by violation,796,2167,2963")
    dat = pd.read_csv(csv, index_col=0)
    ContingencyTable(dat, True)  
    
  elif section == "CA8.2.1: Chi-square goodness of fit test":
    q = 3
    match q:
      case 1:
        pvalue = st.chi2.sf(14.86, 4)
        print(f"{pvalue = :0.3f}")
      case 2:
        print(f"{228*0.25}, {228*0.50}, {228*0.25}")
      case 3:
        dat = pd.DataFrame({
          "Observed": [15, 32, 33],
          "Expected": [20, 40, 20]}, 
          index=["Red", "Pink", "White"])
        Chi2(dat, False)
    
  elif section == "PA8.2.1: Daily distribution of accidents":
    dat = pd.DataFrame({
      "Expected": [150/7]*7,
      "Observed": [14, 32, 20, 22, 20, 24, 18]}, 
      index=["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"])
    Chi2(dat, False)
    
  elif section == "Example 8.2.1: Counting water birds":
    dat = pd.DataFrame({
      "Expected": [0.50, 0.23, 0.12, 0.10, 0.05],
      "Observed": [61, 17, 11, 15, 6]}, 
      index=["Ducks", "Geese", "Cranes", "Swans", "Coots"])
    Chi2(dat, True)
    
  elif section == "Python-Practice 8.1.2: Tukey's HSD":
    from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
    dat = pd.read_csv('ExamScoresGrouped.csv')
    mod = MultiComparison(dat['Scores'], dat['Exam'])
    print(mod.tukeyhsd())
    
  elif section == "Python-Practice 8.1.1: ANOVA":
    # The Exam Score dataset includes scores obtained in 4 exams in a class.
    # Perform a hypothesis test to determine if the mean scores of the exams 
    # are different. Use the 5% level of significance. 
    scores = pd.read_csv('ExamScores.csv')
    # Statistics of each exam
    exam1_score = scores[['Exam1']]
    exam2_score = scores[['Exam2']] 
    exam3_score = scores[['Exam3']] 
    exam4_score = scores[['Exam4']] 
    print(st.f_oneway(exam1_score, exam2_score, exam3_score, exam4_score))

    # imports the necessary libraries
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    # loads the data set
    dat = pd.read_csv('ExamScoresGrouped.csv')
    # builds model and produces aov table
    mod = ols('Scores ~ Exam',data=dat).fit()
    aov_table = sm.stats.anova_lm(mod, typ=2)
    print(aov_table)
    # creates a box plot by exam
    sns.boxplot(x="Exam", y="Scores", data=dat)
    #plt.savefig("examscores.png")
    # Shows the image
    plt.show()



if(__name__ == "__main__"):
  print("Imports loaded!\n")
  Section8()
  print("\nGoodbye!")