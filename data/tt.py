import pandas as pd
from matplotlib import pyplot as plt
df = pd.read_csv('SIN_train_data_with_labels_difTest.csv').iloc[:1,:-1].T

df.plot()
plt.show()