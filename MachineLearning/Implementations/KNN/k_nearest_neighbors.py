import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

points = {'blue':[[2,4],[1,4],[3,1],[3,2],[3,4]],
          'red':[[4,6],[5,7],[7,3],[5,6],[8,5],[10,10]]}

new_point = [3,5]

def eucledian_distance(x,y):
    distance = np.sqrt(np.sum((np.array(x)-np.array(y))**2))
    return distance

class KNearestNeighbors:

    def __init__(self, k=3):
        self.k = k
        self.point = None

    
    def fit(self, points):
        self.points = points

    def predict(self, new_point):
        distances = []
        for category in self.points:
            for point in self.points[category]:
                distance = eucledian_distance(point, new_point)
                distances.append([distance, category])

        categories = [category[1] for category in sorted(distances)[:self.k]]
        # print(categories)
        # print(sorted(distances))
        # result = Counter(categories).most_common(1)
        # print(result)
        result = Counter(categories).most_common(1)[0][0]
        return result

clf = KNearestNeighbors(k=10)
clf.fit(points)
label = clf.predict(new_point)
print(label)

#visualization
ax = plt.subplot()
ax.grid(True,color='#323232')
ax.set_facecolor("black")
ax.figure.set_facecolor('#121212')
ax.tick_params(axis='x', color='white')
ax.tick_params(axis='y', color='white')

for point in points['blue']:
    ax.scatter(point[0],point[1], color='#104DCA', s=60)

for point in points['red']:
    ax.scatter(point[0],point[1], color='#FF0000', s=60)

new_class = clf.predict(new_point)
color = "#FF0000" if new_class == 'red' else "#104DCA"

ax.scatter(new_point[0], new_point[1], marker="*", color = color, s=200, zorder=100)

for point in points['blue']:
    ax.plot([new_point[0], point[0]],[new_point[1],point[1]], color="#104DCA", linestyle="--", linewidth=1)

for point in points['red']:
    ax.plot([new_point[0], point[0]],[new_point[1],point[1]], color="#FF0000", linestyle="--", linewidth=1)

plt.show()