from Basic import *






#==================================================================================================

def Section6():
  section = "Python-Practice 6.1.2: Multiple regression using Python"

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
    
  elif section == "":
    print(f"{0:.3f}")
    
  elif section == "":
    print(f"{0:.3f}")
    
  elif section == "":
    print(f"{0:.3f}")
    
  elif section == "Python-Practice 6.1.2: Multiple regression using Python":
    import statsmodels.formula.api as sms
    fat = pd.read_csv('https://static-resources.zybooks.com/static/fat.csv')
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