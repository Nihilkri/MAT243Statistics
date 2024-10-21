from Basic import *

def Chi2(dat: pd.DataFrame, proportion: bool) -> None:
  if proportion:
    t = dat["Observed"].sum()
    dat["Expected"] *= t
  print(dat)
  print()
  k = len(dat.index)
  df = k - 1
  print(f"{df = }")
  chi2 = sum([(dat["Observed"][row] - dat["Expected"][row]) ** 2 / dat["Expected"][row] for row in dat.index])
  print(f"{chi2 = :0.3f}")


#==================================================================================================

def Section8():
  section = "PA8.2.1: Daily distribution of accidents"

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