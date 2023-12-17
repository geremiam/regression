import numpy as np
import matplotlib.pyplot as plt
import regression

def test_P():
    print(f'{P(11.3, 5) = }')
    print(f'{P(32.6, 20) = }')
    print(f'{P(128., 100) = }')
    print(f'{P(20., 20) = }')
    print(f'{P(26.3, 20) = }')

    return

def test_PolyReg():
    np.set_printoptions(linewidth=1000)
    numpoints = 10
    stdev = 0.1
    x = np.linspace(0., 1., numpoints)
    y = 1. + x + 0.05*x**2 + np.random.normal(0., stdev, size=numpoints)
    y_uncertainty = stdev

    polydeg = {0, 1}

    linreg = regression.PolyReg(x, y, y_uncertainty, polydeg)

    linreg.summary()

    linreg.plot()

    return

def test_PolyReg_2():
    np.set_printoptions(linewidth=1000)
    numpoints = 100
    stdev = np.linspace(0.2, 0.02, numpoints)
    x = np.linspace(0., 1., numpoints)
    y = 1. + x + 0.05*x**2 + np.random.normal(0., stdev, size=numpoints)
    y_uncertainty = stdev

    polydeg = {0, 1, 2}

    linreg = regression.PolyReg(x, y, y_uncertainty, polydeg)

    linreg.summary()

    linreg.plot()

    return

def test_LinReg():
    numpoints = 100
    stdev = 0.1
    x = np.linspace(0., 1., numpoints)
    y = 1. + x + 0.05*x**2 + np.random.normal(0., stdev, size=numpoints)
    y_uncertainty = stdev * np.ones_like(x)

    linreg = regression.LinReg(x, y, y_uncertainty)

    y_pred = linreg.m[0] * x + linreg.c[0]
    normalized_resid = (y - y_pred) / y_uncertainty

    print(f'{linreg.c = }')
    print(f'{linreg.m = }')
    print(f'{linreg.nu = }')
    print(f'{linreg.chisq_min = }')
    print(f'{linreg.P = }')
    print(f'{linreg.D = }')


    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].errorbar(x, y, yerr=y_uncertainty, capsize=2., marker='.', markersize=3., ls='none', zorder=0)
    ax[0].plot(x, y_pred)
    ax[0].set_ylabel(r'$y$')

    ax[1].scatter(x, normalized_resid, marker='.')
    ax[1].set_xlabel(r'$x$')
    ax[1].set_ylabel(r'$(y_i-y_\text{fit}(x_i)) / \sigma_i$')

    plt.show()

def test_compare():
    numpoints = 100
    stdev = 0.1
    x = np.linspace(0., 1., numpoints)
    y = 1. + x + 0.05*x**2 + np.random.normal(0., stdev, size=numpoints)
    y_uncertainty = stdev * np.ones_like(x)

    linreg = regression.LinReg(x, y, y_uncertainty)

    print(f'{linreg.c = }')
    print(f'{linreg.m = }')
    print(f'{linreg.nu = }')
    print(f'{linreg.chisq_min = }')
    print(f'{linreg.P = }')
    print(f'{linreg.D = }')

    polydeg = {0, 1}
    linreg = regression.PolyReg(x, y, y_uncertainty, polydeg)
    linreg.summary()


if __name__=='__main__':
    #test_PolyReg()
    test_PolyReg_2()
    #test_LinReg()
    #test_P()
    #test_compare()
