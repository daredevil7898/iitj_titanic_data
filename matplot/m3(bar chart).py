import matplotlib.pyplot as plt
import numpy as np

categories = np.array(["Meat","Fruit","Milk","Sweets","Vegetables"])
amount = np.array([2,4,5,6,3])

plt.bar(categories,amount,color ="#0c073b")
#plt.barh for horizontal view
plt.title("Bar Chart",fontweight = "bold",fontsize = 20)
plt.xlabel("Food",fontsize = 15)
plt.ylabel("Amount",fontsize = 15)
plt.show()