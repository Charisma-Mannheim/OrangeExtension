import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixing random state for reproducibility
np.random.seed(19680801)
Spectra = pd.read_csv("L:/Promotion/GitHubVerzeichnisse/AddOnsAKWeller/Datensaetze/FTIR_honey_data.csv", sep=';', header=None)
Spectra = Spectra.to_numpy()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

colors = ['r', 'g', 'b', 'y', 'm', 'k']
yticks = [4, 3, 2, 1, 0]
for c, k in zip(colors, yticks):
    # Generate the random data for the y=k 'layer'.
    xs = np.arange(Spectra[k,:].shape[0])
    ys = Spectra[k,:]

    # You can provide either a single color or an array with the same length as
    # xs and ys. To demonstrate this, we color the first bar of each set cyan.
    cs = [c] * len(xs)


    # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
    ax.bar(xs, ys, zs=k, zdir='y', color=cs, alpha=0.8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# On the y axis let's only label the discrete values that we have data for.
ax.set_yticks(yticks)

plt.show()