import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='Times New Roman', size=17)


def GetLevels(array: np.ndarray, rank=2, num=15, sym=False):
    max_ = np.nanpercentile(array, 100 - rank)
    min_ = np.nanpercentile(array, rank)
    if sym:
        max_ = np.max([np.abs(max_), np.abs(min_)])
        min_ = -max_
    levels = np.linspace(min_, max_, num=num)
    return levels


m, n = 121, 31
x = np.arange(int(-m / 2), int(m / 2) + 1)
y = np.arange(int(-n / 2), int(n / 2) + 1)
Y, X = np.meshgrid(y, x)
k = np.loadtxt("./data/041603NASymHeat.txt").reshape((-1, 121, 31))
u, v, f = k[::3], k[1::3], k[2::3]
stepx = 3
stepy = 1
print(np.percentile(np.sqrt(u ** 2 + v ** 2), 50) * 100)
levels = GetLevels(f, rank=5)
for i in range(f.shape[0]):
    plt.figure(figsize=[10, 8])
    m = plt.contourf(x, y, f[i].T, levels=levels, extend='both', cmap="RdBu")

    q = plt.quiver(x[::stepx], y[::stepy], u[i][::stepx, ::stepy].T, v[i][::stepx, ::stepy].T, scale=80)
    plt.colorbar(m, orientation='horizontal', shrink=0.8)
    plt.title("step=%s" % i)
    plt.savefig("./pic/041603NASymHeat/%s.png" % i)

    plt.show()
