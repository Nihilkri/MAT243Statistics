from Basic import *

def ZTest2(sampleMean1:float, sampleMean2:float, popStd1:float, popStd2:float, n1:int, n2:int, tail:int, a:float, sig:int=3):
  """ Section 4.1.1 Two-sample z-test for population means
      The z-test can also be used to determine whether the means of two independent populations are
      the same when the population standard deviations are known. When performing a hypothesis test
      involving the means of two independent populations, the distribution of the z-test statistic
      is assumed to be N(0, sqrt(sigma_1^2 / n_1 + sigma_2^2 / n_2)). In practice, the standard
      deviation for populations are generally unknown, so either the paired or unpaired t-test
      is needed.
      The conditions that must be satisfied are similar to those of a z-test for a population mean.
  """
  stderr = round((popStd1 ** 2 / n1 + popStd2 ** 2 / n2) ** 0.5, sig)
  print(f"{stderr = :.6f}")
  z = round((sampleMean1 - sampleMean2 - 0) / stderr, sig)
  ZTest(z, tail, a, sig)
  
def PairedTTest(sampleMeanDifference:float, hypMeanDifference:float, sampleStd:float, n:int, tail:int, a:float, sig:int=3):
  """ Section 4.1.2 Two-sample t-test
       In a paired t-test or dependent t-test, a sample taken from one population is exposed to two different treatments. The main idea is that measurements are recorded from the same group, usually before and after a treatment is applied or when each of two treatments is applied. Ex: A group of professional cycling athletes is selected for a study on the effects of caffeine dosage on exhaustion times. The populations are the cyclists for each of two dosages. The samples are the measured exhaustion times for each dosage, which implies dependence because the measurements were taken from the same group.
  """
  t = round((sampleMeanDifference - hypMeanDifference) / (sampleStd / n ** 0.5), sig)
  TTest(t, n - 1, tail, a, sig)
  
def UnpairedTTest(sampleMean1:float, sampleMean2:float, hypSampleMean1:float, hypSampleMean2:float, sampleStd1:float, sampleStd2:float, n1:int, n2:int, tail:int, a:float, sig:int=3):
  """ Section 4.1.2 Two-sample t-test
       In an unpaired t-test or independent t-test, a sample taken from one population is not related to a different sample taken from another population. In contrast to the paired t-test, measurements from an unpaired t-test are recorded from different groups when exposed to the same treatment. Ex: The effect of caffeine intake on exhaustion times is studied by measuring the exhaustion times of a randomly selected group of 9 professional cyclists taking caffeine pills and another group of 9 cyclists not taking caffeine pills. The two populations are all cyclists taking caffeine pills and those who are not taking the pills. The samples are the measured exhaustion times from the two groups, each with 9 cyclists, which implies independence because the times are for two different groups of cyclists.
  """
  stderr = round((sampleStd1 ** 2 / n1 + sampleStd2 ** 2 / n2) ** 0.5, sig)
  print(f"{stderr = :.6f}")
  t = round((sampleMean1 - sampleMean2 - (hypSampleMean1 - hypSampleMean2)) / stderr, sig)
  TTest(t, n1 + n2 - 1, tail, a, sig)

def ZTest2Prop(samp1:float, n1:int, samp2:float, n2:int, tail:int, a:float, sig:int=3):
  """ Section 4.2.1 Hypothesis test for the difference between two population proportions
      The z-test can also be used to determine whether the proportions of two distinct populations
      are the same. 
      Where p1 is the probability of success in the first sample, p2 is the probability of success
      in the second sample, p is the overall probability of success when two samples are combined,
      n1 is the size of the first sample, and n2 is the size of the second sample.
  """
  p1 = round(samp1 / n1, sig)
  p2 = round(samp2 / n2, sig)
  p = round((samp1 + samp2) / (n1 + n2), sig)
  if(samp1 < 5):
    print(f"The first proportion is too small. {n1}*{p1} < 5.")
    return 0
  if(n1 - samp1 < 5):
    print(f"The first proportion is too large. {n1}*{p1} > {n1} - 5.")
    return 0
  if(samp2 < 5):
    print(f"The second proportion is too small. {n2}*{p2} < 5.")
    return 0
  if(n2 - samp2 < 5):
    print(f"The second proportion is too large. {n2}*{p2} > {n2} - 5.")
    return 0
  print(f"{p1 = :.{sig}f}")
  print(f"{p2 = :.{sig}f}")
  print(f"p^ = {p:.{sig}f}")
  stderr = (p * (1 - p) * (1 / n1 + 1 / n2)) ** 0.5
  print(f"{stderr = }")
  z = round((p1 - p2 - 0) / stderr, sig)
  ZTest(z, tail, a, sig)
        
#==================================================================================================

def Section4():
  section = "4-4 Response"

  if section == "":
    print(f"{0:.3f}")

  elif section == "4-4 Response":
    ZTest(-1.048, 0, 0.05, 3)
    ZTest(0.5, 0, 0.05, 3)
    ZTest(00, 0, 0.05, 3)

  elif section == "4-4 Discussion":
    from Calculus import np
    print([1, 2, 3] + [4])
    print(np.concatenate([np.array([1, 2, 3]), np.array([4])]))
    dat1 = np.array([
      2.375095026938485, 2.5743268497449394, 2.1787695858587885, 2.5496399367016376, 2.088213282381755,
      3.1390782388863148, 2.255904840042252, 3.0140920529486794, 1.500945082908399, 3.1658620163718387,
      1.759487728399665, 2.9355242950384257, 2.5703111149486713, 2.262466928267701, 3.045180857441924,
      2.6309144792840398, 2.245912479894006, 1.7422111955495443, 2.9986486616499857, 1.6894223317294863,
      2.4105653631242285, 3.1206521148469384, 2.329282140716035, 1.4079987998053123, 2.420572386962487,
      2.6709578565673646, 1.855229864128511, 1.8833175253897338, 1.954341051172483, 2.8621243844976254,
      3.259942674689241, 2.3880490764966242, 2.6358483777668544, 2.921285952150814, 1.8706375381388525,
      3.01405498016116, 2.2671053297632304, 2.2513995681828893, 2.6269320039838977, 2.5218784181982463,
      2.9579669116330747, 3.3748388030879743, 2.246364845377377, 1.8143836553610286, 2.263211082784322,
      2.6274919834620984, 2.2303526769032227, 2.4063025176724477, 2.6160324753821365, 1.5649691989033314
    ])
    dat2 = np.array([
      3.3603522697293493, 1.9388275286311458, 3.0987918836893615, 2.0841825125353495, 1.6375493251110278,
      4.044657840491528, 1.877322428564296, 2.897588378463085, 3.1554606291143266, 1.4278769844409869,
      2.19045243620031, 2.6029449878585007, 2.603477170137309, 2.172476783624875, 2.440130251309244,
      1.8249511195039996, 3.2145566188078183, 3.4718055592695, 3.873943081411059, 1.830489139125068,
      1.9812388842531896, 1.6921172557789466, 1.5137082866479878, 2.484017089907396, 2.349044702352281,
      1.6201424471063373, 2.751433998998887, 3.181778446918358, 2.9425252527534984, 3.5502267050887486,
      2.484748468742191, 1.9541451741491045, 3.0742189758328835, 3.5239971988300782, 2.2589833658924126,
      2.5389181123709155, 3.596263488648441, 3.2635013379648363, 3.1884614991388407, 2.890590916092404,
      2.860631035830941, 2.5708179421637993, 2.8563230805388096, 2.702528914278585, 3.2841456993741946,
      3.2398930096454563, 1.736790302265823, 3.493579726805542, 2.7791544546719438, 2.3358229166985125
    ])
    dat = np.concatenate([dat1, dat2])
    samp1 = len([s for s in dat1 if s < 2.20])
    samp2 = len([s for s in dat2 if s < 2.20])
    n1, mean1, std1 = MeanStd(dat1)
    n2, mean2, std2 = MeanStd(dat2)
    print(f"{mean1 = }")
    print(f"{mean2 = }")
    print(f"{std1 = }")
    print(f"{std2 = }")
    from Module3ConfidenceIntervals import ConfidenceInterval
    print(f"{FormatCI(ConfidenceInterval(0.95, mean1, std1, n1, False))}")
    print(f"{FormatCI(ConfidenceInterval(0.95, mean2, std2, n2, False))}")
    ZTest2Prop(samp1, n1, samp2, n2, 0, 0.05, 3)
    from Calculus import Plot, Const, PDF, Delta, Gaussian
    xs = np.linspace(0, n1, n1)
    ys1 = sorted(dat1)
    ys2 = sorted(dat2)
    f = [
         (xs, ys1, 'r', False),
         (xs, ys2, 'b', False),
         (xs, Const(xs, mean1), 'm', False),
         (xs, Const(xs, mean2), 'c', False),
         (xs, Delta(xs, 24, 5.0), 'w', False),
         (xs, Delta(xs, 25, 5.0), 'w', False),
         (xs, Delta(xs, 26, 5.0), 'w', False)
        ]
    #Plot(f, "Bearing Diameters", "Sample #", "Diameter in cm")
    lx, rx, r = np.min(dat), np.max(dat), 125#1048577
    xs = np.linspace(lx, rx, r)
    ys1 = np.concatenate([np.histogram(ys1, xs, density=True)[0], [0]])
    ys2 = np.concatenate([np.histogram(ys2, xs, density=True)[0], [0]])
    f = [
         (xs, ys1, 'r', False),
         (xs, ys2, 'b', False),
         (xs, Gaussian(xs, mean1, std1, 1.0), 'm', False),
         (xs, Gaussian(xs, mean2, std2, 1.0), 'c', False),
         (xs, Delta(xs, mean1, 1.0), 'm', False),
         (xs, Delta(xs, mean2, 1.0), 'c', False)
        ]
    Plot(f, "Bearing Diameters", "Diameter in cm", "Probability")



  elif section == "CA4.2.1: Hypothesis test for the difference between two population proportions":
    q = 2
    if q == 1:
      ZTest2Prop(245, 800, 304, 1000, 0, 0.01, 3)
    elif q == 2:
      ZTest2Prop(538, 800, 628, 1000, 1, 0.01, 3)

  elif section == "Python-Function 4.2.1: proportions_ztest()":
    from statsmodels.stats.proportion import proportions_ztest
    counts = [95, 125]
    n = [5000, 5000]
    print(proportions_ztest(counts, n))

  elif section == "PA4.2.1: Effectiveness of a vaccine":
    ZTest2Prop(95, 5000, 125, 5000, -1, 0.05, 3)

  elif section == "Example 4.2.2: Adverse reaction to drugs":
    ZTest2Prop(18, 25, 21, 30, 1, 0.01, 3)

  elif section == "Example 4.2.1: Gender and voting":
    ZTest2Prop(70, 132, 63, 105, 0, 0.05, 3)

  elif section == "normalDistributionDefinition":
    from Module24NormalDistribution import normalDistributionDefinition as gauss
    gauss(0, 1)

  elif section == "CA4.1.1: Hypothesis test for the difference between two population means":
    #UnpairedTTest(130, 126, 0, 0, 5, 3, 12, 12, 0, 0.01, 3)
    PairedTTest(84-81, 0, 6, 11, 0, 0.01, 3)

  elif section == "zyDE 4.1.2: Unpaired t-test":
    """ The st.ttest_ind(x, y) command takes two arrays or DataFrame columns x and y as inputs and
        returns the t-statistic and the corresponding two-tailed p-value as outputs.
        The code below conducts the test for the packing machine example above. Both the two-tailed
        test and the one-tailed test are conducted.
    """
    import pandas as pd
    df = pd.read_csv('Machine.csv')
    # Two-tailed test
    print(st.ttest_ind(df['Old'],df['New'],equal_var=False, alternative="two-sided"))
    # One-tailed test
    print(st.ttest_ind(df['Old'],df['New'],equal_var=False, alternative="greater"))
    
  elif section == "zyDE 4.1.1: Paired t-test":
    """ The st.ttest_rel(x, y) function takes two arrays or DataFrame columns x and y with the same
        length as inputs and returns the t-statistic and the corresponding two-tailed p-value as
        outputs.
        The code below loads the exam scores data and uses a paired t-test for the hypotheses
        H_0 : mu_d = 0 and H_a : mu_d != 0 where mu_d is the average difference in students'
        exam 1 and exam 2 scores.
    """
    import pandas as pd
    df = pd.read_csv('ExamScores.csv')
    print(st.ttest_rel(df['Exam1'],df['Exam2']))

  elif section == "Python-Function 4.1.1: ztest(x1, x2)":
    from statsmodels.stats.weightstats import ztest
    sample1 = [21, 28, 40, 55, 58, 60]
    sample2 = [13, 29, 50, 55, 71, 90]
    print(ztest(x1 = sample1, x2 = sample2))
    
  elif section == "PA4.1.1: Commute times":
    ZTest2(5.35, 4.95, 0.5, 0.8, 40, 50, 1, 0.05, 4)

  elif section == "Example 4.1.2: Extrasensory perception":
    ZTest2(12, 10, 4.5, 4.0, 50, 60, 0, 0.05)

  elif section == "Example 4.1.1: Candle burn times":
    ZTest2(27.5, 26.0, 2.5, 3.5, 100, 100, 1, 0.05)


if(__name__ == "__main__"):
  print("Imports loaded!")
  Section4()
  print("Goodbye!")