import matplotlib.pyplot as plt #pie chart
import numpy as np

students = np.array(["Seniors","Freshmen","Sophomores","Juniors"])
numbers = np.array([300,450,249,178])
colors = ["red","yellow","cyan","green"]

plt.pie(numbers,labels=students,autopct="%1.1f%%",colors = colors,explode=[0,0,0.1,0.4],shadow = True,startangle = 120)
plt.show()