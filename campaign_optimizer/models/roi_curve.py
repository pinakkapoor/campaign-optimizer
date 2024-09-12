"""
roi curve modeling for marketing channels

fits diminishing returns curves to historical spend/revenue data.
most channels follow a log or saturating curve — first dollar is worth
more than the millionth dollar.
"""

import numpy as np
from scipy.optimize import curve_fit


class ROICurve:
    """
    models the relationship between spend and revenue for a single channel.

    supports two curve types:
    - 'log': revenue = a * ln(spend + 1) + b
    - 'hill': revenue = max_rev * spend^k / (half_sat^k + spend^k)

    hill (aka adstock saturation) is better for most channels but needs
    more data to fit reliably.
    """

    def __init__(self, curve_type='log'):
        if curve_type not in ('log', 'hill'):
            raise ValueError(f"curve_type must be 'log' or 'hill', got '{curve_type}'")
        self.curve_type = curve_type
        self.params = None
        self._fitted = False

    @staticmethod
    def _log_func(spend, a, b):
        return a * np.log(spend + 1) + b

    @staticmethod
    def _hill_func(spend, max_rev, half_sat, k):
        return max_rev * np.power(spend, k) / (np.power(half_sat, k) + np.power(spend, k))

    def fit(self, spend, revenue):
        """fit the ROI curve to historical data"""
        spend = np.array(spend, dtype=float)
        revenue = np.array(revenue, dtype=float)

        if len(spend) < 3:
            raise ValueError("need at least 3 data points to fit a curve")

        if self.curve_type == 'log':
            popt, _ = curve_fit(self._log_func, spend, revenue, p0=[1, 0], maxfev=5000)
            self.params = {'a': popt[0], 'b': popt[1]}
        else:
            # hill function needs reasonable initial guesses
            p0 = [revenue.max() * 1.5, np.median(spend), 1.0]
            bounds = ([0, 0, 0.1], [np.inf, np.inf, 10])
            popt, _ = curve_fit(self._hill_func, spend, revenue, p0=p0,
                                bounds=bounds, maxfev=10000)
            self.params = {'max_rev': popt[0], 'half_sat': popt[1], 'k': popt[2]}

        self._fitted = True
        return self

    def predict(self, spend):
        """predict revenue for given spend levels"""
        if not self._fitted:
            raise RuntimeError("call fit() first")

        spend = np.array(spend, dtype=float)

        if self.curve_type == 'log':
            return self._log_func(spend, self.params['a'], self.params['b'])
        else:
            return self._hill_func(spend, self.params['max_rev'],
                                   self.params['half_sat'], self.params['k'])

    def marginal_roi(self, spend, delta=1.0):
        """
        marginal ROI at a given spend level.
        this is the key metric for budget allocation — move money from
        channels with low marginal ROI to channels with high marginal ROI.
        """
        if not self._fitted:
            raise RuntimeError("call fit() first")

        spend = np.array(spend, dtype=float)
        rev_at_spend = self.predict(spend)
        rev_at_spend_plus = self.predict(spend + delta)
        return (rev_at_spend_plus - rev_at_spend) / delta

    def summary(self):
        """human-readable summary of the fitted curve"""
        if not self._fitted:
            return "not fitted yet"

        if self.curve_type == 'log':
            return (f"log curve: revenue = {self.params['a']:.2f} * ln(spend + 1) "
                    f"+ {self.params['b']:.2f}")
        else:
            return (f"hill curve: max_rev={self.params['max_rev']:.0f}, "
                    f"half_sat={self.params['half_sat']:.0f}, k={self.params['k']:.2f}")
