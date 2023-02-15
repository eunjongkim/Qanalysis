from abc import abstractmethod, ABC
from typing import Any
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from Qanalysis.helper_functions import UnitfulNumber

class DataAnalysis(ABC):
    """
    Abstract class to implement 
    """

    @abstractmethod
    def analyze(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def plot_result(self) -> None:
        pass

class CurveFitAnalysis(DataAnalysis):
    """
    Class to implement analysis based on `scipy.optimize.curve_fit` function.
    """
    @abstractmethod
    def fit_func(self, x: np.ndarray, *args) -> np.ndarray:
        """
        Fit function to be called during curve_fit. Will be overwritten in subclass.
        """
        pass

    @abstractmethod
    def _guess_init_params(self) -> None:
        """
        Guess initial parameters from data. Will be overwritten in subclass.
        """

    def _set_init_params(self, p0: np.ndarray) -> None:
        if p0 is None:
            self._guess_init_params()
        else:
            self.p0 = p0

    @abstractmethod
    def _save_fit_results(self, popt: np.ndarray, pcov: np.ndarray) -> None:
        pass

    def analyze(
        self, p0: np.ndarray = None, plot: bool = True, **kwargs
    ) -> None:
        """
        Analyze the data with initial parameter specified by `p0`.
        """
        # set initial fit parameters
        self._set_init_params(p0)

        # perform fitting
        popt, pcov = self._curve_fit(**kwargs)
        self.is_analyzed = True

        # save fit results
        self._save_fit_results(popt, pcov)

        if plot:
            self.plot_result()


class CurveFitAnalysis1D(CurveFitAnalysis):
    """
    Base class to implement curve fitting analysis for 1D data
    """
    def __init__(
            self, x: np.ndarray, y: np.ndarray,
            x_name: str = None, y_name: str = None,
            x_unit: str = None, y_unit: str = None
        ) -> None:
        # initialize parameters
        self._x = {'value': x, 'name': x_name, 'unit': x_unit}
        self._y = {'value': y, 'name': y_name, 'unit': y_unit}

        if x_name is None:
            self._x['name'] = 'x'
        if y_name is None:
            self._y['name'] = 'y'

        self.n_pts = len(y)
        self.is_analyzed = False

        # initialize attributes to be overwritten
        self.p0 = None
        self.popt = None
        self.pcov = None
        self.lb = None
        self.ub = None

    def _plot_base(self, fit_n_pts: int = 1000) -> plt.Figure:
        fig = plt.figure()

        x_unitful = UnitfulNumber(
            np.max(np.abs(self._x['value'])),
            base_unit = self._x['unit']
        )
        x_scaler = x_unitful.scaler
        x_scaled_unit = x_unitful.unit

        y_unitful = UnitfulNumber(
            np.max(np.abs(self._y['value'])),
            base_unit = self._y['unit']
        )
        y_scaler = y_unitful.scaler
        y_scaled_unit = y_unitful.unit

        # plot data
        plt.plot(
            self._x['value'] / x_scaler, self._y['value'] / y_scaler,
            '.', label="Data", color="black"
        )

        # plot fit result
        x_fit = np.linspace(self._x['value'][0], self._x['value'][-1], fit_n_pts)

        ## fit based on initial params
        plt.plot(
            x_fit / x_scaler, self.fit_func(x_fit, *(self.p0)) / y_scaler,
            label="Fit (Init. Param.)", lw=2, ls='--', color="orange"
        )
        ## fit based on optimal params with r2 score
        plt.plot(
            x_fit / x_scaler, self.fit_func(x_fit, *(self.popt)) / y_scaler,
            label=f"Fit, $R^{2}$={self.r2_score:.4f}"
        )
        plt.xlabel(f"{self._x['name']} ({x_scaled_unit})")
        plt.ylabel(f"{self._y['name']} ({y_scaled_unit})")
        plt.legend(loc=0, fontsize='x-small')
        return fig

    def _curve_fit(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        if self.lb is not None and self.ub is not None:
            popt, pcov = curve_fit(
                self.fit_func, self._x['value'], self._y['value'],
                p0 = self.p0, bounds = (self.lb, self.ub), **kwargs
            )
        else:
            popt, pcov = curve_fit(
                self.fit_func, self._x['value'], self._y['value'],
                p0=self.p0, **kwargs
            )
        return popt, pcov

    def _save_fit_results(self, popt: np.ndarray, pcov: np.ndarray) -> None:
        self.popt = popt
        self.pcov = pcov
        self.r2_score = r2_score(
            self._y['value'], self.fit_func(self._x['value'], *(self.popt))
        )

    def plot_result(self, fit_n_pts: int = 1000):
        """
        Will be overwritten in subclass
        """
        if not self.is_analyzed:
            raise ValueError("The data must be analyzed before plotting")

        return self._plot_base(fit_n_pts = fit_n_pts)