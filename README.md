Proper polynomial regression, including uncertainties.

The class LinReg has been superseded by PolyReg and will be removed in the future.

Class PolyReg

Constructor: PolyReg(x, y, y_uncertainty, polydeg: set[int])
    x: numpy array of independed variable values
    y: numpy array of depended variable values
    y_uncertainty: float or numpy array of uncertainties of y
    polydeg: set of ints specifying the degrees of the terms to be used in the polynomial fit.
             For example, polydeg={0,1,2} gives f(x) = c0 + c1*x + c2*x^2,
             and polydeg={1} gives f(x) = c1*x.

Attributes:
    params: dictionary of parameter values and uncertainties, with the terms’ degrees serving as keys.
    chisq_min: the minimal value of chi squared, which yields the best fit
    nu: number of degrees of freedom (equal to number of points minus number of fit parameters)
    P: the result of the P-value test for this fit
    D: the result of the Durbin-Watson statistic for this fit
    normalized_resid: the normalized residuals (residuals divided by the corresponding uncertainty)

Methods:
    best_fit(x_input): gives the value of y corresponding to x according to the fit
    summary(): prints a summary of the fit results
    plot(): plots the data, the fit, and the normalized residuals
