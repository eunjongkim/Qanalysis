from Qanalysis.analysis import CurveFitAnalysis1D
from Qanalysis.helper_functions import UnitfulNumber
import matplotlib.pyplot as plt
import numpy as np

class TimeSweep(CurveFitAnalysis1D):
    """
    Base class to implement curve fitting analysis for 1D data plotted against time
    """
    def __init__(
        self, time: np.ndarray, signal: np.ndarray, time_unit: str = 's',
        signal_name: str = 'Signal', signal_unit: str = None
    ):
        super().__init__(
            time, signal, x_name = 'Time', x_unit = time_unit,
            y_name = signal_name, y_unit = signal_unit
        )
        # initialize parameters
        self.time = time
        self.time_unit = time_unit
        self.signal = signal

class AmpSweep(CurveFitAnalysis1D):
    """
    Base class to implement curve fitting analysis for 1D data plotted against amplitude
    """
    def __init__(
        self, amp: np.ndarray, signal: np.ndarray, amp_unit: str = 'V',
        signal_name: str = 'Signal', signal_unit: str = None
    ):
        super().__init__(
            amp, signal, x_name = 'Amplitude', x_unit = amp_unit,
            y_name = signal_name, y_unit = signal_unit
        )
        # initialize parameters
        self.amp = amp
        self.amp_unit = amp_unit
        self.signal = signal