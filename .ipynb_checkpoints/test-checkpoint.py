from matplotlib import pyplot as plt

import seaborn as sns
import numpy as np

#print(sns.get_dataset_names())
tips = sns.load_dataset("tips")
#print(tips.head())
iris = sns.load_dataset("iris")
#print(iris.head())
titanic = sns.load_dataset("titanic")
#print(titanic.head())
planets = sns.load_dataset("planets")
#print(planets.head())


#sns.scatterplot(x="tip", y="total_bill", data=tips, hue="day", size="size", palette="YlGnBu")

#sns.histplot(tips['tip'], kde=True, bins=30)

#sns.boxplot(x="day", y="tip", data=tips, hue="sex", palette="YlGnBu")

#sns.stripplot(x="day", y="tip", data=tips, hue="sex", palette="YlGnBu", dodge=True)

#sns.jointplot(x="tip", y="total_bill", data=tips, kind="hex", cmap="YlGnBu")

#sns.pairplot(titanic.select_dtypes(['number']), hue='pclass')

sns.heatmap(titanic.select_dtypes(include=['number']).corr(), annot=True, cmap="YlGnBu")

plt.show()
"""
# Generate 100 random data points along 3 dimensions
x, y, scale = np.random.randn(3, 100)
fig, ax = plt.subplots()

# Map each onto a scatterplot we'll create with Matplotlib
ax.scatter(x=x, y=y, c=scale, s=np.abs(scale)*500)
ax.set(title="Some random data, created with JupyterLab!")
plt.show()
"""