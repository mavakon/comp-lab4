import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import DBSCAN

# Зчитування датасету
data_df = pd.read_csv("DS5.txt", sep=" ", header=None, names=["x", "y"])

# Ділення на зв'язані області
clustering = DBSCAN(eps=1, min_samples=1).fit(data_df)
data_df['label'] = clustering.labels_

# Обчислення центрів ваги
centroids = data_df.groupby('label').mean().values

# Створення фігури з розміром полотна 960x540 пікселів
fig = plt.figure(figsize=(960/100, 540/100))

# Встановлення осей графіка, щоб вони займали весь простір фігури
ax = fig.add_axes([0, 0, 1, 1])

# Запис переліку областей
unique_labels = data_df['label'].unique()

# Утворення колірної мапи для областей
cmap = plt.cm.get_cmap('cool', len(unique_labels))

# Відображення точок усіх зв'язаних областей
for i, label in enumerate(unique_labels):
    cluster_points = data_df[data_df['label'] == label]
    ax.scatter(cluster_points['x'], cluster_points['y'], color=cmap(i))

# Відображення центрів ваги
ax.scatter(centroids[:, 0], centroids[:, 1], color='red', s=2.5**2)

# Збереження результату у файл output_1.png
plt.savefig("output_1.png", dpi=100)

# Відображення полотна (і видалення фігури при закритті вікна)
plt.show()

# Створення фігури з розміром полотна 960x540 пікселів
fig = plt.figure(figsize=(960/100, 540/100))

# Встановлення осей графіка, щоб вони займали весь простір фігури
ax = fig.add_axes([0, 0, 1, 1])

# Побудова діаграми Вороного
vor = Voronoi(centroids)
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=2, point_size=5)

# Відображення точок датасету
ax.scatter(data_df['x'], data_df['y'], color=(0.9, 0.9, 0.9))

# Збереження результату у файл output_2.png
plt.savefig("output_2.png", dpi=100)

# Відображення полотна
plt.show()
