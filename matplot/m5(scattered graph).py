import matplotlib.pyplot as plt
import numpy as np


x = np.array([2,3,4,5,6,7,8,9])
y = np.array([45,67,88,45,34,76,89,90])
d = dict(fontsize = 20,fontweight = "bold")
plt.title("Graph",d)
plt.xlabel("Hours Studied",d)
plt.ylabel("Marks",d)
plt.scatter(x,y,color = "blue",alpha = 0.4,label = "Class A")#alpha is transparency
plt.grid()
plt.legend()
plt.show()