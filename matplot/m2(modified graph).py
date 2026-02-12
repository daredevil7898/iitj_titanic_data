import matplotlib.pyplot as plt
import numpy as np

x = np.array([2023,2024,2025,2026])
y1 = np.array([14,25,65,19])
y2 = np.array([29,25,91,10])

line_style = dict(marker = "o",markersize = 10,markerfacecolor = "Black",markeredgecolor = "Red",color = "Blue",linestyle = "solid",linewidth = 2)

label = dict(fontsize = 20,family = "Arial",fontweight = "bold",color = "black")


plt.title("Graph",label)
plt.xlabel("Year",label)
plt.ylabel("Units Purchased",label)
plt.plot(x,y1,**line_style)
plt.plot(x,y2,**line_style)



plt.xticks(x)#for to remove unnecessary decimal values
plt.tick_params(axis = "both",
                color = "red")

#for bg color
ax = plt.gca()
ax.set_facecolor("white")


plt.grid() # adds lines to the graph,could be used for both axis or just only for x axis or y axis


plt.show()