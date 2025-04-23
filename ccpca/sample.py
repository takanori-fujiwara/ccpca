import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing

from cpca import CPCA
from ccpca import CCPCA

dataset = datasets.load_wine()

X = dataset.data
y = dataset.target
X = preprocessing.scale(X)

cpca = CPCA()

# auto alpha selection
cpca.fit(fg=X[y == 0], bg=X[y != 0])
# manual alpha selection
# cpca.fit(fg=X[y == 0], bg=X[y != 0], auto_alpha_selection=False, alpha=2.15)

X_r = cpca.transform(X)


colors = ["navy", "turquoise", "darkorange"]
lw = 2
plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], [0, 1, 2]):
    plt.scatter(
        X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=0.8, lw=lw, label=target_name
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title(f"cPCA of IRIS dataset (alpha={cpca.get_best_alpha()})")
plt.show()

# ccPCA with automatic alpha selection
ccpca = CCPCA()
X_r2 = ccpca.fit_transform(X[y == 0], X[y != 0])

plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], [0, 1, 2]):
    plt.scatter(
        X_r2[y == i, 0],
        X_r2[y == i, 1],
        color=color,
        alpha=0.8,
        lw=lw,
        label=target_name,
    )
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title(f"ccPCA of IRIS dataset (alpha ={ccpca.get_best_alpha()})")
plt.show()
