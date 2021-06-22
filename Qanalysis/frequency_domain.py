# class FrequencyDomain:
    
# def simple_resonance_detector()

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from .helper_functions import *


def lorentzian_fit_func(f, f0, gamma, a, b):
    omega, omega0 = 2 * np.pi * f, 2 * np.pi * f0

    return a + b / np.pi * (gamma / 2) / ((omega - omega0) ** 2 + (gamma / 2) ** 2)

def analyze_lorentzian(f, sig, p0=None):
    sig_mag = sig
    if sig.dtype == complex:
        sig_mag = np.abs(sig) ** 2

    if p0 is None:
        if (np.max(sig_mag) - np.mean(sig_mag)) > (np.mean(sig_mag) - np.min(sig_mag)):
            # peak detected case
            f0 = f[np.argmax(sig_mag)]
            a = np.mean(sig_mag[np.argsort(sig_mag)[:int(len(sig_mag) // 10)]]) # baseline (average of smallest 10% samples)
            # linewidth is extracted from sample closest to half-max
            gamma = 2 * np.abs(f[np.argmin(np.abs(sig_mag - 0.5 * (np.max(sig_mag) + a)))] - f0)
            b = np.pi * gamma / 2 * (np.max(sig_mag) - a)
            
            p0 = [f0, gamma, a, b]
        elif (np.max(sig_mag) - np.mean(sig_mag)) < (np.mean(sig_mag) - np.min(sig_mag)):
            # valley detected case
            f0 = f[np.argmin(sig_mag)]
            a = np.mean(sig_mag[np.argsort(-sig_mag)[:int(len(sig) // 10)]]) # baseline (average of largest 10% samples)
            # linewidth is extracted from sample closest to half-max
            gamma = 2 * np.abs(f[np.argmin(np.abs(sig_mag - 0.5 * (np.min(sig_mag) + a)))] - f0)
            b = np.pi * gamma / 2 * (np.min(sig_mag) - a)
        
            p0 = [f0, gamma, a, b]

    fit = curve_fit(lorentzian_fit_func, f, sig_mag, p0=p0, 
                    bounds=([p0[0] * 0.5, p0[1] * 0.5, 0, p0[3] * 0.1], 
                            [p0[0] * 1.5, p0[1] * 1.5, np.inf, p0[3] * 10]))
    return fit

def gaussian_fit_func(f, f0, a, c, d):
    return a * np.exp(-(f - f0)**2 / (2 * c**2)) + d

def analyze_gaussian(f, sig, p0=None):
    sig_mag = sig
    if sig.dtype == complex:
        sig_mag = np.abs(sig) ** 2
        
    if p0 is None:
        if (np.max(sig_mag) - np.mean(sig_mag)) > (np.mean(sig_mag) - np.min(sig_mag)):
            # peak detected case
            f0 = f[np.argmax(sig_mag)]
            d = np.mean(sig_mag[np.argsort(sig_mag)[:int(len(sig_mag) // 10)]]) # baseline (average of smallest 10% samples)
            # linewidth is extracted from sample closest to half-max
            c = 1 / np.sqrt(2) * np.abs(f[np.argmin(np.abs(sig_mag - ((np.max(sig_mag) - d) / np.exp(1) + d)))] - f0)
            a = (np.max(sig_mag) - d)
            
            p0 = [f0, a, c, d]
        elif (np.max(sig_mag) - np.mean(sig_mag)) < (np.mean(sig_mag) - np.min(sig_mag)):
            # valley detected case
            f0 = f[np.argmin(sig_mag)]
            d = np.mean(sig_mag[np.argsort(-sig_mag)[:int(len(sig) // 10)]]) # baseline (average of largest 10% samples)
            # linewidth is extracted from sample closest to half-max
            c = 1 / np.sqrt(2) * np.abs(f[np.argmin(np.abs(sig_mag - ((np.min(sig_mag) - d) / np.exp(1) + d)))] - f0)
            a = (np.min(sig_mag) - d)
        
            p0 = [f0, a, c, d]
        
        fit = curve_fit(gaussian_fit_func, f, sig_mag, p0=p0,
                        bounds=([p0[0] * 0.5, p0[1] * 0.5, p0[2] * 0.1, 0], 
                                [p0[0] * 1.5, p0[1] * 1.5, p0[2] * 10, np.inf]))
    return fit
    

# class DispersiveShift:
    
    
class TimeDomain:
    def __init__(self, freq, signal):
        # initialize parameters
        self.frequency = freq
        self.signal = signal
        self.n_pts = len(self.signal)
        self.is_analyzed = False
        self.p0 = None
        self.popt = None
        self.pcov = None

    def _init_fit_params(self, df):
        # to be overwritten in subclasses
        pass

    def _guess_init_params(self):
        """
        Guess initial parameters from data. Will be overwritten in subclass
        """

    def _set_init_params(self, p0):
        if p0 is None:
            self._guess_init_params()
        else:
            self.p0 = p0
    def _save_fit_results(self, popt, pcov):
        self.popt = popt
        self.pcov = pcov

    def analyze(self, p0=None, plot=True, **kwargs):
        """
        Analyze the data with initial parameter `p0`.
        """
        # set initial fit parameters
        self._set_init_params(p0)
        # perform fitting
        popt, pcov = curve_fit(self.fit_func, self.frequency, self.signal,
                               p0=self.p0, **kwargs)
        self.is_analyzed = True

        # save fit results
        self._save_fit_results(popt, pcov)

        if plot:
            self.plot_result()

    def _plot_base(self):
        fig = plt.figure()

        # plot data
        plt.plot(self.time / 1e-6, self.signal, '.', label="Data", color="black")
        plt.xlabel(r"Time ($\mu$s)")
        plt.ylabel("Signal")
        plt.legend(loc=0, fontsize=14)

        fig.tight_layout()
        return fig

    def plot_result(self):
        """
        Will be overwritten in subclass
        """
        if not self.is_analyzed:
            raise ValueError("The data must be analyzed before plotting")


class acStarkShift:
    """
    Class to implement analysis of ac Stark shift measurement with varying
    power applied to the readout resonator.
    
    Parameters
    ----------
    freq : `numpy.ndarray`
        A 1D numpy array that specifies the range of frequencies (in units of
        Hz) for qubit spectroscopy.
    power_dBm : `numpy.ndarray`
        A 1D numpy array that specifies the range of power (in units of dBm)
        for readout resonator.
    signal : `numpy.ndarray`
        A 2D numpy array that stores the complex signal obtained from qubit
        spectroscopy with taken with a range of readout power. The row indices
        and column indices correspond to indices of `power_dBm` and `freq`,
        respectively.
    disp_freq_shift : `float`
        Dispersive frequency shift of readout resonator 2 * chi / (2 * pi) in 
        units of Hz.
    
    Attributes
    ----------
    
    
    Methods
    -------
    
    """
    def __init__(self, freq, power_dBm, signal, disp_freq_shift):
        self.frequency = freq
        # complex signal obtained from spectroscopy
        self.signal = signal
        # readout power in units of dBm
        self.power_dBm = power_dBm
        # dispersive frequency shift (2*chi / 2*pi) in units of Hz
        self.disp_freq_shift = disp_freq_shift # 2 * chi / (2 * pi)
        
        self.p0 = None
        self.popt = None
        self.pcov = None

    def analyze(self, plot=True, p0=None):
        
        self.res_frequency = np.zeros(len(self.power_dBm))
        for idx in range(len(self.power_dBm)):
            lorentzian_fit = analyze_lorentzian(self.frequency, self.signal[idx])
            self.res_frequency[idx] = lorentzian_fit[0][0]

        self._set_init_params(p0)
        
        self.popt, self.pcov = curve_fit(self.fit_func, self.power_dBm,
                                         self.res_frequency, p0=self.p0)
        a, f0 = self.popt

        self.single_photon_power_dBm = watt_to_dBm(self.disp_freq_shift / a)
        
        if plot:
            self.plot_result()

    def _set_init_params(self, p0):
        if p0 is None:
            dp = dBm_to_watt(np.max(self.power_dBm))
            maxind, minind = np.argmax(self.power_dBm), np.argmin(self.power_dBm)
            df = self.res_frequency[maxind] - self.res_frequency[minind]
            self.p0 = [df / dp, self.res_frequency[np.argmin(self.power_dBm)]]
        else:
            self.p0 = p0
        
    def fit_func(self, power_dBm, a, f0):
        
        power_Watt = dBm_to_watt(power_dBm)
        return a * power_Watt + f0
        
    def plot_result(self):
        fig = plt.figure()
        plt.pcolormesh(self.frequency / 1e9, self.power_dBm, np.abs(self.signal), shading='auto')
        plt.plot(self.res_frequency / 1e9, self.power_dBm, '.', color='k', label='Res. Freq.')
        plt.plot(self.fit_func(self.power_dBm, *self.popt) / 1e9,
                 self.power_dBm, '--', color='white', label='Fit')
        plt.title(r"Single-photon power = $%.2f$ dBm" % self.single_photon_power_dBm)
        plt.legend()
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Power (dBm)")
        