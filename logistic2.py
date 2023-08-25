from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

iris=datasets.load_iris()
# iris dataset is related for leaves(length width etc)


x = iris["data"][:, 3:]
#here 2 reffers to verginica flower
#astype converts that values to 1 if true ,0 if false (b/c regression did not  understand true false..)
y = (iris["target"] == 2).astype(np.int)
print(x)
clf=LogisticRegression()
clf.fit(x,y)
clf.predict(x,y)
#testing for 2.6 either belong to that category or not
example=clf.predict(([2.6]))
print(example)

x_new=np.linspace(0,3,1000).reshape(-1,1) #gives the 1000 points under 0 to 3 ----for x axis
print(x_new)
y_prob=clf.predict_proba(x_new)  # probabilities

plt.plot(x_new,y_prob,"g-")
plt.show()


