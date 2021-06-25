# class FrequencyDomain:
    
# def simple_resonance_detector()

from scipy.optimize import curve_fit, least_squares
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

    
class FrequencyDomain:
    def __init__(self, freq, signal):
        # initialize parameters
        self.frequency = freq
        self.signal = signal
        self.n_pts = len(self.signal)
        self.is_analyzed = False
        self.p0 = None
        self.popt = None
        self.pcov = None

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
        _, self.frequency_prefix = number_with_si_prefix(np.max(np.abs(self.frequency)))
        self.frequency_scaler = si_prefix_to_scaler(self.frequency_prefix)

        plt.plot(self.frequency / self.frequency_scaler,
                 self.signal, '.', label="Data", color="black")
        plt.xlabel("Frequency (" + self.frequency_prefix + "Hz)")
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

    def _get_const_baseline(self, baseline_portion=0.2,
                            baseline_ref='symmetric'):
        
        samples = self.signal
        
        N = len(samples)
        if baseline_ref == 'left':
            bs = np.mean(samples[:int(baseline_portion * N)])
        elif baseline_ref == 'right':
            bs = np.mean(samples[int(-baseline_portion * N):])
        elif baseline_ref == 'symmetric':
            bs_left = np.mean(samples[int(-baseline_portion * N / 2):])
            bs_right = np.mean(samples[:int(baseline_portion * N / 2)])
            bs = np.mean([bs_left, bs_right])

        return bs

class LorentzianFit(FrequencyDomain):
    """
    
    """
    def fit_func(self, f, f0, df, a, b):
        """
        Lorentzian fit function

        Parameters
        ----------
        f : TYPE
            DESCRIPTION.
        f0 : TYPE
            Resonant frequency.
        df : TYPE
            Full-width half-maximum linewidth.
        a : TYPE
            DESCRIPTION.
        b : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        return a / ((f - f0) ** 2 + (df / 2) ** 2) + b
    
    def _guess_init_params(self):
        """
        Guess initial parameters from data.
        """
        signal = self.signal
        f = self.frequency

        b0 = self._get_const_baseline()
        
        peak_A0, dip_A0 = np.max(signal) - b0, np.min(signal) - b0
        if peak_A0 > - dip_A0: # peak detected case
            A0 = peak_A0
            f0 = f[np.argmax(signal)]
        else: # valley detected case
            A0 = dip_A0
            f0 = f[np.argmin(signal)]

        # linewidth is extracted from sample closest to half-max(arg1, arg2, _args)
        df0 = 2 * np.abs(f[np.argmin(np.abs(signal - (0.5 * A0 + b0)))] - f0)
        a0 = A0 * (df0 / 2) ** 2

        self.p0 = [f0, df0, a0, b0]

    def _save_fit_results(self, popt, pcov):
        super()._save_fit_results(popt, pcov)

        self.f0 = popt[0]
        self.f0_sigma_err = np.sqrt(pcov[0, 0])
        self.df = popt[1]
        self.df_sigma_err = np.sqrt(pcov[1, 1])

    def plot_result(self, fit_n_pts=1000):
        super().plot_result()

        # get most of the plotting done
        fig = self._plot_base()

        freq_fit = np.linspace(self.frequency[0],
                               self.frequency[-1], fit_n_pts)

        plt.plot(freq_fit / self.frequency_scaler,
                 self.fit_func(freq_fit, *(self.p0)),
                 label="Fit (Init. Param.)", ls='--', lw=2, color="orange")

        plt.plot(freq_fit / self.frequency_scaler,
                 self.fit_func(freq_fit, *(self.popt)),
                 label="Fit (Opt. Param.)", lw=2, color="red")
        
        f0_string = (r"$f_0$ = %.4f $\pm$ %.4f " %
                     (self.f0 / self.frequency_scaler,
                      2 * self.f0_sigma_err / self.frequency_scaler) +
                     self.frequency_prefix + 'Hz')

        _, df_prefix = number_with_si_prefix(np.max(np.abs(self.df)))
        df_scaler = si_prefix_to_scaler(df_prefix)

        df_string = (r"$\Delta f$ = %.4f $\pm$ %.4f " %
                     (self.df / df_scaler, 2 * self.df_sigma_err / df_scaler) +
                     df_prefix + 'Hz')

        plt.title(f0_string + ', ' + df_string)
        plt.legend(loc=0, fontsize='x-small')
        fig.tight_layout()
        plt.show()

        return fig

class GaussianFit(FrequencyDomain):

    def fit_func(self, f, f0, sigma_f, a, b):
        
        return a * np.exp(-(f - f0) ** 2 / (2 * sigma_f ** 2)) + b

    def _guess_init_params(self):
        """
        Guess initial parameters from data.
        """
        signal = self.signal
        f = self.frequency

        b0 = self._get_const_baseline()

        peak_a0, dip_a0 = np.max(signal) - b0, np.min(signal) - b0
        if peak_a0 > - dip_a0: # peak detected case
            a0 = peak_a0
            f0 = f[np.argmax(signal)]
        else: # valley detected case
            a0 = dip_a0
            f0 = f[np.argmin(signal)]

        # sigma linewidth is extracted from sample closest to 1/2 of max
        sigma_f0 = np.sqrt(np.log(2) / 2) * np.abs(f[np.argmin(np.abs(signal - (0.5 * a0 + b0)))] - f0)

        self.p0 = [f0, sigma_f0, a0, b0]

    def _save_fit_results(self, popt, pcov):
        super()._save_fit_results(popt, pcov)

        self.f0 = popt[0]
        self.f0_sigma_err = np.sqrt(pcov[0, 0])
        self.sigma_f = popt[1]
        self.sigma_f_sigma_err = np.sqrt(pcov[1, 1])

    def plot_result(self, fit_n_pts=1000):
        super().plot_result()

        # get most of the plotting done
        fig = self._plot_base()

        freq_fit = np.linspace(self.frequency[0],
                               self.frequency[-1], fit_n_pts)

        plt.plot(freq_fit / self.frequency_scaler,
                 self.fit_func(freq_fit, *(self.p0)),
                 label="Fit (Init. Param.)", ls='--', lw=2, color="orange")

        plt.plot(freq_fit / self.frequency_scaler,
                 self.fit_func(freq_fit, *(self.popt)),
                 label="Fit (Opt. Param.)", lw=2, color="red")

        f0_string = (r"$f_0$ = %.4f $\pm$ %.4f " %
                     (self.f0 / self.frequency_scaler,
                      2 * self.f0_sigma_err / self.frequency_scaler) +
                     self.frequency_prefix + 'Hz')

        _, sigma_f_prefix = number_with_si_prefix(np.max(np.abs(self.sigma_f)))
        sigma_f_scaler = si_prefix_to_scaler(sigma_f_prefix)

        sigma_f_string = (r"$\sigma_f$ = %.4f $\pm$ %.4f " %
                     (self.sigma_f / sigma_f_scaler,
                      2 * self.sigma_f_sigma_err / sigma_f_scaler) +
                     sigma_f_prefix + 'Hz')

        plt.title(f0_string + ', ' + sigma_f_string)
        plt.legend(loc=0, fontsize='x-small')

        fig.tight_layout()
        plt.show()

        return fig


# class SingleSidedS11Fit(FrequencyDomain):
#     """
#     Class for implementing fitting of lineshape in reflection measurement
#     """
#     def __init__(self, freq, data, fit_type='complex'):
#         super().__init__(freq, data, df=df, fit_mag_dB=fit_mag_dB, plot_mag_dB=plot_mag_dB)
        
#         self.fit_type = fit_type

#     def fit_func(self, *args):
#         """
#         Reflection fit for resonator single-sided coupled to waveguide
#         according to relation
#         S11 = a * exp(i(ϕ + 2pi * (f - f[0]) * τ)) .* ...
#              (1 - 2 * QoverQe * (1 + 2i * δf / f0) / (1 + 2i * Q * (f - f0) / f0))

#         Note: If df = 0 (no background) this gives
#         S11 = 1 - (2 kappa_e) / (kappa_i + kappa_e + 2i (omega - omega0))
#             = (kappa_i - kappa_e + 2i (omega - omega0)) / (kappa_i + kappa_e + 2i (omega - omega0))
#         See Aspelmeyer et al, "Cavity Optomechanics", Rev. Mod. Phys. 86, 1391 (2014).
        
#         If the fit type is complex, the arguments of `fit_func` are:
#             f, f0, Q, QoverQe, df, a, phi, tau
#         If the fit type is magnitude, the arguments of `fit_func` are:
#             f, f0, Q, QoverQe, df, a
#         If the fit type is phase, the arguments of `fit_func` are:
#             f, f0, Q, QoverQe, df, phi, tau
#         """
#         if self.fit_type == 'complex':
#             f, f0, Q, QoverQe, df, a, phi, tau = args

#             return (a * np.exp(1j * (phi + 2 * np.pi * (f - f[0]) * tau)) *
#                     (1 - 2 * QoverQe * (1 + 2j * df / f0) /
#                     (1 + 2j * Q * (f - f0) / f0)))
#         elif self.fit_type == 'magnitude':
#             f, f0, Q, QoverQe, df, a = args

#             return np.abs(a * (1 - 2 * QoverQe * (1 + 2j * df / f0) /
#                                (1 + 2j * Q * (f - f0) / f0)))
#         elif self.fit_type == 'phase':
#             f, f0, Q, QoverQe, df, phi, tau = args

#             return np.angle(np.exp(1j * (phi + 2 * np.pi * (f - f[0]) * tau)) *
#                             (1 - 2 * QoverQe * (1 + 2j * df / f0) /
#                              (1 + 2j * Q * (f - f0) / f0)))

#     def _estimate_f0_FWHM(self):
#         f = self.frequency
#         mag2 = np.abs(self.data) ** 2
        
#         # background in magnitude estimated by linear interpolation of first and last point
#         mag2_bg = (mag2[-1] - mag2[0]) / (f[-1] - f[0]) * (f - f[0]) + mag2[0]
        
#         mag2_subtracted = mag2 - mag2_bg
#         f0 = f[mag2_subtracted.argmin()]
#         smin, smax = np.min(mag2_subtracted), np.max(mag2_subtracted)

#         # data in frequency < f0 or frequency >= f0
#         f_l, s_l = f[f < f0], mag2_subtracted[f < f0]
#         f_r, s_r = f[f >= f0], mag2_subtracted[f >= f0]

#         # find frequencies closest to the mean of smin and smax
#         f1 = f_l[np.abs(s_l - 0.5 * (smin + smax)).argmin()]
#         f2 = f_r[np.abs(s_r - 0.5 * (smin + smax)).argmin()]

#         # numerically find full width half max from magnitude squared
#         Δf = f2 - f1

#         return f0, Δf

#     def _guess_init_params(self, df):

#         # magnitude data
#         _mag = np.abs(self.data)
#         # phase data
#         _ang = np.angle(self.data)
#         # unwrapped phase data
#         _angU = np.unwrap(_ang)

#         f = self.frequency

#         a0 = np.sqrt((_mag[0] ** 2 + _mag[-1] ** 2) / 2)
#         phi0 = _angU[0]
#         tau0 = 0.0
#         if (np.max(_angU) - np.min(_angU)) > 2.1 * np.pi:
#             # if phase data at start and stop frequencies differ by more than 2pi,
#             # perform phase subtraction associated with delay
#             tau0 = (_angU[-1] - _angU[0]) / ((f[-1] - f[0]))/ (2 * np.pi)

#         # Estimate total Q from the FWHM in |mag|^2
#         f0, Δf = self._estimate_f0_FWHM()
#         amin = _mag[np.abs(f - f0).argmin()]
#         QoverQe0 = 0.5 * (1 - amin / a0)
#         Q0 = f0 / Δf
        
#         if self.fit_type == 'complex':
#             self.p0 = [f0, Q0, QoverQe0, 0.0, a0, phi0, tau0]
#         elif self.fit_type == 'magnitude':
#             self.p0 = [f0, Q0, QoverQe0, 0.0, a0]
#         elif self.fit_type == 'phase':
#             self.p0 = [f0, Q0, QoverQe0, 0.0, phi0, tau0]
    

# class acStarkShift:
#     """
#     Class to implement analysis of ac Stark shift measurement with varying
#     power applied to the readout resonator.
    
#     Parameters
#     ----------
#     freq : `numpy.ndarray`
#         A 1D numpy array that specifies the range of frequencies (in units of
#         Hz) for qubit spectroscopy.
#     power_dBm : `numpy.ndarray`
#         A 1D numpy array that specifies the range of power (in units of dBm)
#         for readout resonator.
#     signal : `numpy.ndarray`
#         A 2D numpy array that stores the complex signal obtained from qubit
#         spectroscopy with taken with a range of readout power. The row indices
#         and column indices correspond to indices of `power_dBm` and `freq`,
#         respectively.
#     disp_freq_shift : `float`
#         Dispersive frequency shift of readout resonator 2 * chi / (2 * pi) in 
#         units of Hz.
    
#     Attributes
#     ----------
    
    
#     Methods
#     -------
    
#     """
#     def __init__(self, freq, power_dBm, signal, disp_freq_shift):
#         self.frequency = freq
#         # complex signal obtained from spectroscopy
#         self.signal = signal
#         # readout power in units of dBm
#         self.power_dBm = power_dBm
#         # dispersive frequency shift (2*chi / 2*pi) in units of Hz
#         self.disp_freq_shift = disp_freq_shift # 2 * chi / (2 * pi)
        
#         self.p0 = None
#         self.popt = None
#         self.pcov = None

#     def analyze(self, plot=True, p0=None):
        
#         self.res_frequency = np.zeros(len(self.power_dBm))
#         for idx in range(len(self.power_dBm)):
#             lorentzian_fit = analyze_lorentzian(self.frequency, self.signal[idx])
#             self.res_frequency[idx] = lorentzian_fit[0][0]

#         self._set_init_params(p0)
        
#         self.popt, self.pcov = curve_fit(self.fit_func, self.power_dBm,
#                                          self.res_frequency, p0=self.p0)
#         a, f0 = self.popt

#         self.single_photon_power_dBm = watt_to_dBm(self.disp_freq_shift / a)
        
#         if plot:
#             self.plot_result()

#     def _set_init_params(self, p0):
#         if p0 is None:
#             dp = dBm_to_watt(np.max(self.power_dBm))
#             maxind, minind = np.argmax(self.power_dBm), np.argmin(self.power_dBm)
#             df = self.res_frequency[maxind] - self.res_frequency[minind]
#             self.p0 = [df / dp, self.res_frequency[np.argmin(self.power_dBm)]]
#         else:
#             self.p0 = p0
        
#     def fit_func(self, power_dBm, a, f0):
        
#         power_Watt = dBm_to_watt(power_dBm)
#         return a * power_Watt + f0
        
#     def plot_result(self):
#         fig = plt.figure()
#         plt.pcolormesh(self.frequency / 1e9, self.power_dBm, np.abs(self.signal), shading='auto')
#         plt.plot(self.res_frequency / 1e9, self.power_dBm, '.', color='k', label='Res. Freq.')
#         plt.plot(self.fit_func(self.power_dBm, *self.popt) / 1e9,
#                  self.power_dBm, '--', color='white', label='Fit')
#         plt.title(r"Single-photon power = $%.2f$ dBm" % self.single_photon_power_dBm)
#         plt.legend()
#         plt.xlabel("Frequency (GHz)")
#         plt.ylabel("Power (dBm)")
        

# class WaveguideCoupledS21Fit(FrequencyDomain):
#     """
#     Class for imlementing fitting of lineshape in reflection measurement
#     """
#     def __init__(self, freq, data, df=0, fit_mag_dB=False, plot_mag_dB=False):
#         super().__init__(freq, data, df=df, fit_mag_dB=fit_mag_dB, plot_mag_dB=plot_mag_dB)

#         self.fit_type = "WaveguideCoupledTransmission"

#     def fit_func(self, f, f0_MHz, Q, QoverQe, δf, a, ϕ, τ_ns):
#         """
#         Reflection fit for resonator single-sided coupled to waveguide
#         according to relation
#         S11 = a0 * exp(i(ϕ0 + 2pi * (f - f[0]) * τ0)) .* ...
#              (1 - QoverQe * (1 + 2i * δf / f0) / (1 + 2i * Q * (f - f0) / f0))

#         Note: If df = 0 (no background) this gives
#         S21 = 1 - (kappa_e) / (kappa_i + kappa_e + 2i (omega - omega0))
#             = (kappa_i + 2i (omega - omega0)) / (kappa_i + kappa_e + 2i (omega - omega0))
#         See Khalil et al, "An analysis method for asymmetric resonator
#         transmission applied to superconducting devices",
#         J. Appl. Phys. 111, 054510 (2012).
#         """
#         return (a * np.exp(1j * (ϕ + 2 * np.pi * (f - f[0]) * τ_ns * 1e-9)) *
#                 (1 - QoverQe * (1 + 2j * δf / f0_MHz) /
#                 (1 + 2j * (Q) * (f/1e6 - f0_MHz) / (f0_MHz))))

#     def _init_fit_params(self, df):
#         # magnitude data
#         _mag = np.abs(self.data)
#         # phase data
#         _ang = np.angle(self.data)
#         # unwrapped phase data
#         _angU = np.unwrap(_ang)

#         f = self.frequency

#         a0 = _mag[0]
#         ϕ0 = _angU[0]
#         τ0 = 0.0
#         if (np.max(_angU) - np.min(_angU)) > 2.1 * np.pi:
#             # if phase data at start and stop frequencies differ more than 2pi,
#             # perform phase subtraction associated with delay
#             τ0 = (_angU[-1] - _angU[0]) / ((f[-1] - f[0]))/ (2 * np.pi)

#         # Estimate total Q from the FWHM in |mag|^2
#         f0, Δf = self._estimate_f0_FWHM()
#         QoverQe0 = (1 - np.min(_mag) / a0)
#         Q0 = f0 / Δf
#         p0_mag, p0_ang = self._prepare_fit_params(f0, Q0, QoverQe0,
#                                                   df, a0, ϕ0, τ0)

#         self.p0 = p0_mag + p0_ang
#         return p0_mag, p0_ang