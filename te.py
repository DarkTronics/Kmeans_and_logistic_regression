import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('adult.csv')
df = pd.DataFrame(data)

print(df)

sns.scatterplot(data=df, x='age', y='hours-per-week', hue='income', style='capital-gain')
plt.title('Scatter plot with more than 2 dimensions')
plt.show()
