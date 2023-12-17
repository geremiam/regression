''' Module with regression functionalities '''
import numpy as np
from scipy.special import loggamma
from scipy.integrate import quad
import matplotlib.pyplot as plt

def P(chisq_min: float, nu: int) -> float:
    ''' P-value test: the probability that chi_sq is greater than chisq_min. '''
    def X(chisq: float) -> float:
        ''' The probability distribution for chi_sq '''
        # Need to calculate log to avoid overflow
        log_X = (nu/2.-1.)*np.log(chisq) - chisq/2. - (nu/2.)*np.log(2.) - loggamma(nu/2.)
        return np.exp(log_X)
    return quad(X, chisq_min, np.inf)

def calculate_D(normalized_resid):
    ''' The Durbin-Watson statistic for the fit '''
    numerator   = np.sum( (normalized_resid[1:] - normalized_resid[:-1])**2 )
    denominator = np.sum( normalized_resid**2 )
    D = numerator / denominator
    return D


class LinReg:
    ''' Performs linear regression of a 1D function.
        Allows for uncertainty on the dependent variable.
    '''
    def __init__(self, x, y, y_uncertainty):
        ''' x: array of independent variables
            y: array of dependent variables
            y_uncertainty: float or array of uncertainties on the dependent variables
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


class PolyReg:
    ''' Performs polynomial regression of a 1D function.
        Allows for uncertainty on the dependent variable.
    '''
    def __init__(self, x, y, y_uncertainty, polydeg: set[int]):
        ''' x: array of independent variables
            y: array of dependent variables
            y_uncertainty: array of uncertainties on the dependent variables
            polydeg: set of polynomial degrees desired in the fit
        '''
        # Need to keep track of order of coefficients
        polydeg = sorted(list(polydeg))

        # Number of degrees of freedom
        self.nu = x.size - len(polydeg)
        
        # Save copies
        self.x = x
        self.y = y
        self.y_uncert = y_uncertainty
        self.polydeg = polydeg

        # Calculate params
        self.params = self._calculate_params()
        # Calculate residuals
        self.normalized_resid = (self.y - self.best_fit(self.x)) / self.y_uncert
        # Minimum chisq
        self.chisq_min = np.sum( self.normalized_resid**2 )
        # The Durbin-Watson statistic for the fit
        self.D = calculate_D(self.normalized_resid)
        # The P value for the fit
        self.P = P(self.chisq_min, self.nu)


    def _calculate_params(self):
        ''' Calculate fit parameters with proper uncertainties. '''

        w = self.y_uncert**-2

        # The optimal parameters are the solution to the linear equation
        # A @ params = gamma,
        # where the matrix A and the vector gamma are as defined below

        gamma = np.array( [w * self.y * self.x**deg for deg in self.polydeg] ).sum(axis=-1)

        A = np.array( [[w * self.x**(deg1+deg2) for deg1 in self.polydeg] for deg2 in self.polydeg] ).sum(axis=-1)

        # Give a warning if A has negative eigenvalues
        # These seem to happen because of rounding error
        A_eigvals = np.linalg.eigvalsh(A)
        A_eigvals_nonpos = A_eigvals[A_eigvals<=0.]
        if A_eigvals_nonpos.size > 0:
            print(f'*** Warning *** The matrix A has non-positive eigenvalues {A_eigvals_nonpos}')

        params_values = np.linalg.solve(A, gamma)

        # The uncertainties are as follows
        params_uncert = np.sqrt(np.diag(np.linalg.inv(A)))

        params = {self.polydeg[i] : (params_values[i], params_uncert[i]) for i in range(len(self.polydeg))}

        return params

    def best_fit(self, x_input):
        ''' Calculate correspding values for x_input according to best fit '''
        x_input = np.asarray(x_input)

        y_pred = np.zeros_like(x_input)

        for deg in self.params:
            y_pred += self.params[deg][0] * x_input**deg

        return y_pred

    def summary(self):
        print('\tSUMMARY')

        print('\n\tFIT PARAMETERS')
        for deg in self.params:
            print(f'\tc{deg} = {self.params[deg]}')

        print('\n\tMINIMAL CHI SQUARED AND DEGREES OF FREEDOM')
        print(f'\tchisq_min = {self.chisq_min}')
        print(f'\tnu = {self.nu}')

        print('\n\tP-VALUE TEST')
        print(f'\tP = {self.P}')

        print('\n\tDURBIN-WATSON STATISTIC')
        print(f'\tD = {self.D}')


    def plot(self):
        ''' Plot the data, fit and residuals '''
        fig, ax = plt.subplots(2, 1, sharex=True)

        ax[0].errorbar(self.x, self.y, yerr=self.y_uncert, capsize=2., marker='.', markersize=3., ls='none', zorder=0)
        ax[0].plot(self.x, self.best_fit(self.x))
        ax[0].set_ylabel(r'$y$')
    
        ax[1].scatter(self.x, self.normalized_resid, marker='.')
        ax[1].set_xlabel(r'$x$')
        ax[1].set_ylabel('normalized residuals')
    
        plt.show()
        
        return
