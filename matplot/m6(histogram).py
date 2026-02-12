import matplotlib.pyplot as plt #Histogram
import numpy as np

scores = np.random.normal(loc = 80,scale = 10,size = 100)
scores = np.clip(scores,0,100)

plt.title("Exam Score",size = 20)
plt.xlabel("Scores")
plt.ylabel("No. of Students")
plt.hist(scores,color = "green",edgecolor ="black")
plt.show()


f = np.array([3,4,8])
g = np.array([7,8,9])

print(f+g)