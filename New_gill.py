import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import functools
import time

m, n = 121, 31
an = 1.e-5
eps = 1.e-5


def run_time(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kw):
        start = time.time()
        res = fn(*args, **kw)
        print('%s 运行了 %f 秒' % (fn.__name__, time.time() - start))
        return res

    return wrapper


def bc(Grid: np.ndarray):
    """
    x方向边界条件
    :param Grid: np.ndarray
    :return: Grid
    """
    Grid[0] = Grid[1]
    Grid[-1] = Grid[-2]
    return Grid


def bc1(func):
    """
    边界条件的修饰器，便于后面使用边界条件
    :param func: 需要修饰的函数
    :return: func
    """

    def wrapper(*args, **kw):
        Grid = func(*args, **kw)
        Grid = bc(Grid)
        return Grid

    return wrapper


@bc1  # 空间平滑需要边界条件
def smooth(Grid: np.ndarray, sms=0.5, point=9):
    """
    空间平滑函数
    :param Grid: np.ndarray, 需要平滑的二维数组
    :param sms: 空间平滑系数
    :param point: 平滑方式 9：九点平滑；5：五点平滑
    :return: 平滑后的数组
    """
    if point == 9:
        wi = np.array([
            [sms ** 2 / 4, sms / 2 * (1 - sms), sms ** 2 / 4],
            [sms / 2 * (1 - sms), 1 - 2 * sms * (1 - sms) - sms ** 2, sms / 2 * (1 - sms)],
            [sms ** 2 / 4, sms / 2 * (1 - sms), sms ** 2 / 4]
        ])
    else:
        wi = np.array([
            [0, sms / 4, 0],
            [sms / 4, 1 - sms, sms / 4],
            [0, sms / 4, 0]
        ])
    wi = np.rot90(wi, 2)
    Grid1 = np.concatenate([Grid[:, [0]], Grid, Grid[:, [-1]]], axis=1)
    Grid[1:-1, ] = signal.convolve2d(Grid1, wi, mode="valid")
    return Grid


def smoth(sms=0.5, point=9):
    """
    空间平滑修饰器函数
    :param sms: 空间平滑系数
    :param point: 平滑方式 9：九点平滑；5：五点平滑
    :return: 平滑函数
    """

    def smooth_sp(func):
        def wrapper(*args, **kwargs):
            Grid = func(*args, **kwargs)
            Grid = smooth(Grid, sms=sms, point=point)
            return Grid

        return wrapper

    return smooth_sp


@bc1
def px_c(gx: np.ndarray, ar: np.ndarray):
    """
    x 方向上的中央差 , 使用x边界修饰器对其修饰
    :param gx: d/dx
    :param ar: 求差分的数组
    :return: gx
    """
    gx[1:-1] = ar[2:] - ar[:-2]
    return gx


@bc1
def py_c(gy: np.ndarray, ar: np.ndarray):
    """
    y 方向上的中央差 , 使用x边界修饰器对其修饰
    :param gy: d/dy
    :param ar: 求差分的数组
    :return: gy
    """
    gy[:, 1:-1] = ar[:, 2:] - ar[:, :-2]
    gy[:, 0] = ar[:, 1] - ar[:, 0]
    gy[:, -1] = ar[:, -1] - ar[:, -2]
    return gy


def pt(gt: np.ndarray, ar0: np.ndarray, ar1: np.ndarray, step: int):
    """
    时间差分函数
    :param gt: d/dt
    :param ar0: 前一时刻值
    :param ar1: 后一时刻值
    :param step: 时间步数
    :return: gt
    """
    gt[:] = (ar1 - ar0) / step
    return gt


@smoth()
@bc1
def divg(div: np.ndarray, ax: np.ndarray, ay: np.ndarray,
         dx: float, dy: float, px: np.ndarray, py: np.ndarray):
    """
    求 散度的函数
    :param div: 散度
    :param ax: u
    :param ay: v
    :param dx: x方向步长
    :param dy: y方向步长
    :param px: du/dx
    :param py: dv/dy
    :return: div
    """
    div[:] = px_c(px, ax) / dx / 2 + py_c(py, ay) / dy / 2
    return div


def trans(a: np.ndarray, b: np.ndarray, c: np.ndarray,
          u: np.ndarray, v: np.ndarray, w: np.ndarray):
    """
    赋值函数，将数组之间的值快速转换
    :param a:
    :param b:
    :param c:
    :param u:
    :param v:
    :param w:
    :return: None
    """
    u[:] = a
    v[:] = b
    w[:] = c


def smoth_time(a0: np.ndarray, a1: np.ndarray, a2: np.ndarray, alp=0.25):
    """
    空间平滑函数
    :param a0: 0 时刻 数组
    :param a1: 1 时刻 数组
    :param a2: 2 时刻 数组
    :param alp: 平滑系数
    :return: a1
    """
    a1[:] = a1 + alp * (a2 + a0 - a1 * 2)
    return a1


def grd_all(dx: float, dy: float, u1: np.ndarray, v1: np.ndarray, f1: np.ndarray,
            u0: np.ndarray, v0: np.ndarray, f0: np.ndarray,
            Q: np.ndarray, Y: np.ndarray, dtu: np.ndarray,
            dtv: np.ndarray, dtf: np.ndarray, div: np.ndarray,
            gpx: np.ndarray, gpy: np.ndarray, px: np.ndarray, py: np.ndarray):
    """
    求 du/dt ,dv/dt , df/dt
    :param dx: float
    x方向步长
    :param dy: float
    y方向步长
    :param u1: u1
    :param v1: v1
    :param f1: f1
    :param u0: 上一时刻 u
    :param v0: 上一时刻 v
    :param f0: 上一时刻 f
    :param Q: 加热函数
    :param Y: 网格Y值
    :param dtu: du/dt
    :param dtv: dv/dt
    :param dtf: df/dt
    :param div: 散度
    :param gpx: df/dx
    :param gpy: df/dy
    :param px: du/dx
    :param py: dv/dy
    :return: None
    """
    divg(div, u1, v1, dx, dy, px, py)
    dtf[:] = -Q - div - an * f0
    dtu[:] = 1 / 2 * Y * v1 - px_c(gpx, f1) / dx - eps * u0
    dtv[:] = -1 / 2 * Y * u1 - py_c(gpy, f1) / dy - eps * v0


@run_time
def integral0():
    """
    主积分函数
    :return: None
    """
    # 初始化变量
    u0, u1, u2 = np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, n))
    v0, v1, v2 = np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, n))
    f0, f1, f2 = np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, n))
    dtu, dtv, dtf, div, gpx, gpy = np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, n)), np.zeros((m, n)), np.zeros(
        (m, n)), np.zeros((m, n))
    px = np.zeros(div.shape)
    py = np.zeros(div.shape)
    # 初始化网格
    x = np.arange(int(-m / 2), int(m / 2) + 1)
    y = np.arange(int(-n / 2), int(n / 2) + 1)
    # 设置加热源
    Y, X = np.meshgrid(y, x)
    L = 20
    Q = Y * np.cos(np.pi / 2 / L * X) * np.exp(-1 / 15 * Y ** 2)
    Q[np.abs(X) > L] = 0
    # 绘制热源
    plt.contourf(x, y, Q.T, cmap="Reds")
    plt.savefig("Q.png")
    plt.show()
    # 积分初始化
    ii = 0  # 积分序号
    dt0 = 0.03  # 时间步长
    # 时间前差、中央差控制变量
    iBack = 0
    # 空间步长
    dx, dy = 1, 1
    # 截止时间
    end_step = 3000
    f_save = open("./data/041603NASymHeat.txt", "ab")
    # 开始积分
    while ii < end_step:
        if np.mod(ii, 24) == 0:  # 时间前差起步，然后每24步进行两次前差
            dt = dt0
            if iBack == 0:
                trans(u1, v1, f1, u0, v0, f0)
            iBack += 1
        else:  # 中央差
            dt = dt0 * 2
        # 向前积分
        grd_all(dx, dy, u1, v1, f1, u0, v0, f0, Q, Y, dtu, dtv, dtf, div, gpx, gpy, px, py)
        u2, v2, f2 = u0 + dt * dtu, v0 + dt * dtv, f0 + dt * dtf
        # 设置边界条件
        bc(u2), bc(v2), bc(f2)
        # 每23步平滑一波
        if np.mod(ii, 23) == 0:
            smooth(u2), smooth(v2), smooth(f2)
        # 中央差时时间平滑
        if dt != dt0:
            u1 = smoth_time(u0, u1, u2)
            v1 = smoth_time(v0, v1, v2)
            f1 = smoth_time(f0, f1, f2)
        # 时间变量转换
        if np.mod(ii, 24) == 0:  # 前差时
            trans(u2, v2, f2, u1, v1, f2)
        else:  # 中央差时
            trans(u1, v1, f1, u0, v0, f0)
            trans(u2, v2, f2, u1, v1, f1)
        # 前差两步结束
        if iBack == 2:
            iBack = 0
        if iBack == 0:
            ii += 1
            if np.mod(ii, 30) == 0:
                np.savetxt(f_save, u1, fmt="%.04f")
                np.savetxt(f_save, v1, fmt="%.04f")
                np.savetxt(f_save, f1, fmt="%.04f")


if __name__ == '__main__':
    integral0()
