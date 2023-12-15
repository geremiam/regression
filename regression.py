''' Module with regression functions '''
import numpy as np
from scipy.special import gamma
from scipy.special import loggamma
from scipy.integrate import quad

def P(chisq_min: float, nu: int) -> float:
    ''' P-value test: the probability that chi_sq is greater than chisq_min. '''
    def X(chisq: float) -> float:
        ''' The probability distribution for chi_sq '''
        # Need to calculate log to avoid overflow
        log_X = (nu/2.-1.)*np.log(chisq) - chisq/2. - (nu/2.)*np.log(2.) - loggamma(nu/2.)
        return np.exp(log_X)
    return quad(X, chisq_min, np.inf)

class LinReg:
    ''' Performs linear regression of a 1D function.
        Allows for uncertainty on the dependent variable.
    '''
    def __init__(self, x, y, y_uncertainty):
        ''' x: array of independent variables
            y: array of dependent variables
            y_uncertainty: array of uncertainties on the dependent variables
        '''
        # Number of degrees of freedom
        self.nu = x.size - 2 # There are two parameters in a linear fit

        # The fit parameters
        self.c, self.m = self._calculate_params(x, y, y_uncertainty)

        # Calculate the residuals
        normalized_resid = self._calculate_residuals(x, y, y_uncertainty)

        # The Durbin-Watson statistic for the fit
        self.D = self._calculate_D(normalized_resid)

        # The best-fit value of chisq
        self.chisq_min = self._calculate_chisq_min(normalized_resid)

        # The P value for the fit
        self.P = P(self.chisq_min, self.nu)

    def _calculate_params(self, x, y, y_uncertainty):
        ''' Calculate slope and intercept with proper uncertainties.
            See Sect. 6.3 of Hughes & Hase.
        '''

        w = y_uncertainty**-2

        sum_w   = np.sum(w)
        sum_wx  = np.sum(w*x)
        sum_wy  = np.sum(w*y)
        sum_wxx = np.sum(w*x*x)
        sum_wxy = np.sum(w*x*y)

        Delta = sum_w * sum_wxx - sum_wx**2

        c_val = ( sum_wxx*sum_wy - sum_wx*sum_wxy ) / Delta
        m_val = ( sum_w*sum_wxy - sum_wx*sum_wy ) / Delta

        c_uncertainty = np.sqrt( sum_wxx / Delta )
        m_uncertainty = np.sqrt( sum_w / Delta )

        return (c_val, c_uncertainty), (m_val, m_uncertainty)

    def _calculate_residuals(self, x, y, y_uncertainty):
        y_pred = self.m[0] * x + self.c[0]
        normalized_resid = (y - y_pred) / y_uncertainty
        return normalized_resid

    def _calculate_chisq_min(self, normalized_resid):
        chisq_min = np.sum( normalized_resid**2 )
        return chisq_min

    def _calculate_D(self, normalized_resid):
        numerator   = np.sum( (normalized_resid[1:] - normalized_resid[:-1])**2 )
        denominator = np.sum( normalized_resid**2 )
        D = numerator / denominator
        return D


def test_P():
    print(f'{P(11.3, 5) = }')
    print(f'{P(32.6, 20) = }')
    print(f'{P(128., 100) = }')
    print(f'{P(20., 20) = }')
    print(f'{P(26.3, 20) = }')

def test_LinReg():
    numpoints = 100
    stdev = 0.05
    x = np.linspace(0., 10., numpoints)
    y = 1. + x + 0.001*x**2 + np.random.normal(0., stdev, size=numpoints)
    y_uncertainty = stdev * np.ones_like(x)

    linreg = LinReg(x, y, y_uncertainty)
    
    y_pred = linreg.m[0] * x + linreg.c[0]
    normalized_resid = (y - y_pred) / y_uncertainty

    print(f'{linreg.c = }')
    print(f'{linreg.m = }')
    print(f'{linreg.nu = }')
    print(f'{linreg.chisq_min = }')
    print(f'{linreg.P = }')
    print(f'{linreg.D = }')

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].errorbar(x, y, yerr=y_uncertainty, capsize=2., marker='.', markersize=3., ls='none', zorder=0)
    ax[0].plot(x, y_pred)
    ax[0].set_ylabel(r'$y$')

    ax[1].scatter(x, normalized_resid, marker='.')
    ax[1].set_xlabel(r'$x$')
    ax[1].set_ylabel(r'$(y_i-y_\text{fit}(x_i)) / \sigma_i$')
    
    plt.show()

if __name__=='__main__':
    test_LinReg()
    #test_P()
