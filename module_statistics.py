import numpy as np
import statistics
from scipy import stats
from sklearn import linear_model
from scipy.optimize import fsolve


def norm_dist(x, t, print_values=False):
    """ Vrne vrednosti funkcije za normalno porazdelitev.
    :param x = vse vrednosti vzorca,
    :param t = srednje vrednosti intervalov,
    :param print_values = izpiše vrednosti stand. deviacije in srednjo vrednost mi"""

    sigma = statistics.stdev(x)
    mi = statistics.mean(x)

    f = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((t - mi) ** 2) / (2 * sigma ** 2))
    F = stats.norm.cdf(t, mi, sigma)

    if print_values:
        print("sigma = ", sigma)
        print("mi = ", mi)
    else:
        pass

    return f, F


def exp_dist(x, t, print_values=False):
    """ Vrne vrednosti funkcije za eksponentno porazdelitev.
    :param x = vse vrednosti vzorca,
    :param t = srednje vrednosti intervalov,
    :param print_values = izpiše vrednosti stand. deviacije, srednji čas MTTF in lambdo"""

    MTTF = statistics.mean(x)
    lambd = 1 / MTTF
    sigma = np.sqrt(1 / lambd ** 2)

    f = lambd * np.exp(-lambd * t)

    F = 1 - np.exp(-lambd * t)

    if print_values:
        print("sigma = ", sigma)
        print("MTTF = ", MTTF)
        print("lambda = ", lambd)
    else:
        pass

    return f, F


def Weibull_dist(x, t, print_values=False):
    """ Vrne vrednosti funkcije za eksponentno porazdelitev.
    :param x = vse vrednosti vzorca,
    :param t = zgornje meje intervalov,
    :param print_values = izpiše vrednosti beta in theta"""

    print('Po kateri metodi želiš izračunati parametra beta in theta? \n'
          'Možnost: \n'
          '0 - metoda medialnih rangov (MMR) \n'
          '1 - metoda največjega verjetja (MLE)')

    N = len(x)
    method = input()
    if method == '0':   # MMR
        x_sorted = np.sort(x)

        F = []
        for i in range(N):
            F.append((i + 1 - 0.3) / (N + 0.4))
        F = np.array(F)

        # linearna regresija
        index = np.arange(len(x_sorted))
        for i in range(len(x_sorted)):
            index[i] = 0
        X = np.stack((index, np.log(x_sorted)))
        transpose = np.transpose(X)
        reg = linear_model.LinearRegression()
        reg.fit(transpose, np.log(-np.log(1 - F)))
        k = reg.coef_[1]
        n = reg.intercept_
        print("k = ", k)
        print("n = ", n)
        beta = k
        theta = np.exp(-n / beta)

    elif method == '1':     # MLE

        def func(beta):
            return -1 / beta - 1 / N * np.sum(np.log(x)) + (np.sum(x ** beta * np.log(x))) / np.sum(x ** beta)

        print("Ugibaj začetno vrednost!")
        start_val = input()                 # potrebno vpisati začetno vrednost za numerično računanje
        start_val = float(start_val)        # če napiše napako za deljenje z 0 je potrebno poizkusiti z drugo vrednostjo

        beta = fsolve(func, x0=start_val)
        theta = np.average(x ** beta) ** (1 / beta)

    else:
        print("Napačen vnos!")

    f = beta / theta * (t / theta) ** (beta - 1) * np.exp(-((t / theta) ** beta))
    F = 1 - np.exp(-(t / theta) ** beta)

    if print_values:
        print("beta = ", beta)
        print("theta = ", theta)

    return f, F
