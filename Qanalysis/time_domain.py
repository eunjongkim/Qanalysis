import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.signal import windows
from typing import Optional
from .helper_functions import number_with_si_prefix, si_prefix_to_scaler
from scipy.linalg import svd


def _get_envelope(s, t, f):
    """
    Find the envelope of an oscillating real-valued signal `s` as a function of time `t` by demodulating at the
    frequency specified by `f` followed by low-pass filtering at cutoff of half the specified frequency.
    """
    # time step
    dt = t[1] - t[0]

    # perform manual demodulation to get envelope
    I = s * np.cos(2 * np.pi * f * t)
    Q = - s * np.sin(2 * np.pi * f * t)

    # extract envelope by low-pass filtering at cutoff of f / 2
    _window = int(1 / (f * dt))
    _hann = windows.hann(_window * 2, sym=True)
    _hann = _hann / np.sum(_hann)
    envComplex = np.convolve(I + 1j * Q, _hann, 'same') * 2
    return envComplex[:len(s)]

class TimeDomain:
    def __init__(self, time, signal):
        # initialize parameters
        self.time = time
        self.signal = signal
        self.n_pts = len(self.signal)
        self.is_analyzed = False
        self.p0 = None
        self.popt = None
        self.pcov = None
        self.lb = None
        self.ub = None

    def fit_func(self):
        """
        Fit function to be called during curve_fit. Will be overwritten in subclass
        """

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
        if self.lb is not None and self.ub is not None:
            popt, pcov = curve_fit(self.fit_func, self.time, self.signal,
                                   p0=self.p0, bounds=(self.lb, self.ub),
                                   **kwargs)
        else:
            popt, pcov = curve_fit(self.fit_func, self.time, self.signal,
                                   p0=self.p0, **kwargs)
        self.is_analyzed = True

        # save fit results
        self._save_fit_results(popt, pcov)

        if plot:
            self.plot_result()

    def _plot_base(self):
        fig = plt.figure()

        # plot data
        _, self.time_prefix = number_with_si_prefix(np.max(np.abs(self.time)))
        self.time_scaler = si_prefix_to_scaler(self.time_prefix)

        # plot data
        plt.plot(self.time / self.time_scaler, self.signal, '.',
                 label="Data", color="black")
        plt.xlabel("Time (" + self.time_prefix + 's)')
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

class Rabi(TimeDomain):
    """
    Class to analyze and visualize Rabi oscillation data
    """

    def __init__(self, time, signal):
        super().__init__(time, signal)
        self.TRabi = None
        self.RabiFrequency = None

    def fit_func(self, t, TRabi, F, t0, Td, a, b, c):
        """
        The strongly-driven Rabi oscillation curve based on Torrey's solution
        a e^(-t/Td) + b e^(-t/TRabi) cos(2π F(t-t0)) + c

        1/Td = 1/T2 - (1/T2 - 1/T1) Δ²/(Δ² + Ω₀²) + O(α³)
        1/TRabi = 1/T2 - 1/2(1/T2 - 1/T1) Ω₀²/(Δ² + Ω₀²) + O(α³)
        Ω = √(Δ² + Ω₀²) + O(α²)
        where α ≡ (1 / T₂ - 1 / T₁) / Ω₀ ≪ 1 is a small parameter when damping
        rates (1/T₂, 1/T₁) are very small compared to the Rabi frequency Ω₀=2πF₀.
        """
        return a * np.exp(- t / Td) + b * np.exp(- t / TRabi) * np.cos(2 * np.pi * F * (t - t0)) + c

    def _guess_init_params(self):
        # perform fft to find frequency of Rabi oscillation
        freq = np.fft.rfftfreq(len(self.signal), d=(self.time[1] - self.time[0]))
        # initial parameter for Rabi frequency from FFT
        F0 = freq[np.argmax(np.abs(np.fft.rfft(self.signal - np.mean(self.signal))))]

        a0 = np.max(np.abs(self.signal - np.mean(self.signal)))
        b0 = 0.0
        c0 = np.mean(self.signal)
        t0 = 0.0
        T0 = self.time[-1] - self.time[0]

        self.p0 = [T0, F0, t0, T0, a0, b0, c0]

    def _save_fit_results(self, popt, pcov):
        super()._save_fit_results(popt, pcov)

        self.TRabi = popt[0]
        self.RabiFrequency = popt[1]

    def plot_result(self, fit_n_pts=1000):
        super().plot_result()

        fig = self._plot_base()

        time_fit = np.linspace(self.time[0], self.time[-1], fit_n_pts)

        plt.plot(time_fit / 1e-6, self.fit_func(time_fit, *(self.popt)), label="Fit")
        plt.title(r"$T_R = %.5f \mu\mathrm{s}$, $\Omega_R/2\pi = %.5f \mathrm{MHz}" % (self.TRabi / 1e-6,
                                                                                       self.OmegaRabi / (2 * np.pi * 1e6)))
        fig.tight_layout()
        plt.show()

class RamseyWithVirtualZRotation(TimeDomain):
    def __init__(self, phase, signal):
        # initialize parameters
        self.phase = phase
        self.signal = signal
        self.n_pts = len(self.signal)
        self.is_analyzed = False
        self.p0 = None
        self.popt = None
        self.pcov = None

        self.amplitude = None
        self.phase_offset = None

        self.amplitude_err = None
        self.phase_offset_err = None

    def fit_func(self, phi, phi0, a, b):
        """
        Ramsey fringes generated by two pi/2 pulses with relative phase phi
        on the second pulse.
        """
        return a * np.cos(phi - phi0) + b
    
    def _guess_init_params(self):
        b0 = np.mean(self.signal)
        signal0 = self.signal - b0
        a0 = np.max(np.abs(signal0))
        phi0 = - np.arccos(signal0[0] / a0) + self.phase[0]
        
        self.p0 = [phi0, a0, b0]

    def _save_fit_results(self, popt, pcov):
        super()._save_fit_results(popt, pcov)
        self.phase_offset = popt[0]
        self.amplitude = popt[1]
        self.phase_offset_err = np.sqrt(pcov[0, 0])
        self.amplitude_err = np.sqrt(pcov[1, 1])
        
    def analyze(self, p0=None, plot=True, **kwargs):
        """
        Analyze Ramsey fringes with virtual Z rotation curve with model
        """
        self._set_init_params(p0)
        lb = [-np.inf, 0, np.min(self.signal)]
        ub = [np.inf, (np.max(self.signal) - np.min(self.signal)),
              np.max(self.signal)]
        popt, pcov = curve_fit(self.fit_func, self.phase, self.signal,
                               p0=self.p0, **kwargs,
                               bounds=(lb, ub))
        
        self.is_analyzed = True

        self._save_fit_results(popt, pcov)

        if plot:
            self.plot_result()

    def plot_result(self, fit_n_pts=1000):
        super().plot_result()

        fig = plt.figure()

        # plot data
        plt.plot(self.phase, self.signal, '.', label="Data", color="black")
        plt.xlabel(r"Phase (rad)")
        plt.ylabel("Signal")
        plt.legend(loc=0, fontsize=14)

        phase_fit = np.linspace(self.phase[0], self.phase[-1], fit_n_pts)

        plt.plot(phase_fit, self.fit_func(phase_fit, *(self.popt)),
                 label="Fit", lw=2, color="red")
        plt.title(r"amplitude = %.5f" % self.amplitude)
        fig.tight_layout()
        plt.show()

class PowerRabi(TimeDomain):
    def __init__(self, amp, signal):
        # initialize parameters
        self.amp = amp
        self.signal = signal
        self.n_pts = len(self.signal)
        self.is_analyzed = False
        self.p0 = None
        self.popt = None
        self.pcov = None

        self.amp_pi = None    # pi pulse amplitude
        self.amp_pi2 = None   # pi/2 pulse amplitude

    def fit_func(self, amp, amp_pi, a, b):
        """
        Rabi oscillation curve with fixed pulse duration and only driving amplitude swept
        a / 2 (1 - cos(π * amp / amp_pi)) + b
        Here amp_pi is the amplitude corresponding to the pi-pulse.
        """
        return a / 2 * (1 - np.cos(np.pi * amp / amp_pi)) + b

    def _guess_init_params(self):
        # perform fft to find frequency of Rabi oscillation
        freq = np.fft.rfftfreq(len(self.signal), d=(self.amp[1] - self.amp[0]))
        
        sig0 = self.signal - np.mean(self.signal)
        # initial parameter for Rabi frequency from FFT
        F0 = freq[np.argmax(np.abs(np.fft.rfft(sig0)))]
        dF = freq[1] - freq[0]
        amp_pi0 = 1 / (2 * F0)
        b0 = self.signal[0]
        a0 = np.max(self.signal) - np.min(self.signal)
        if np.abs(b0 - np.min(self.signal)) > np.abs(b0 - np.max(self.signal)):
            a0 *= -1

        self.p0 = [amp_pi0, a0, b0]
        if a0 > 0:
            self.lb = [1 / (2 * (F0 + dF)), 0.5 * a0, -np.inf]
            self.ub = [1 / (2 * max(F0 - dF, dF / 2)), np.inf, np.inf]
        else:
            self.lb = [1 / (2 * (F0 + dF)), -np.inf, -np.inf]
            self.ub = [1 / (2 * max(F0 - dF, dF / 2)), 0.5 * a0, np.inf]

    def _save_fit_results(self, popt, pcov):
        super()._save_fit_results(popt, pcov)
        self.amp_pi = popt[0]
        self.amp_pi2 = self.amp_pi / 2
        self.amp_pi_sigma_err = np.sqrt(pcov[0, 0])
        
    def analyze(self, p0=None, plot=True, **kwargs):
        """
        Analyze Power Rabi oscillation curve with model
        """
        self._set_init_params(p0)

        popt, pcov = curve_fit(self.fit_func, self.amp, self.signal,
                               p0=self.p0, bounds=(self.lb, self.ub),
                               **kwargs)
        self.is_analyzed = True

        self._save_fit_results(popt, pcov)

        if plot:
            self.plot_result()

    def plot_result(self, fit_n_pts=1000):
        super().plot_result()

        fig = plt.figure()

        # plot data
        plt.plot(self.amp, self.signal, '.', label="Data", color="black")
        plt.xlabel(r"Amplitude (A.U.)")
        plt.ylabel("Signal")

        amp_fit = np.linspace(self.amp[0], self.amp[-1], fit_n_pts)
        plt.plot(amp_fit, self.fit_func(amp_fit, *(self.p0)),
                 label="Fit (init. param.)", lw=2, ls='--', color="orange")
        plt.plot(amp_fit, self.fit_func(amp_fit, *(self.popt)),
                 label="Fit (opt. param.)", lw=2, color="red")
        plt.title(r"$a_{\pi} = %.5f \pm %.5f$" %
                  (self.amp_pi, 2 * self.amp_pi_sigma_err))
        plt.legend(loc=0, fontsize='x-small')
        fig.tight_layout()
        plt.show()

class PopulationDecay(TimeDomain):
    """
    Class to analyze and visualize population decay (T1 experiment) data.
    """
    def __init__(self, time, signal):
        super().__init__(time, signal)
        self.T1 = None
        self.T1_sigma_err = None

    def fit_func(self, t, T1, a, b):
        """
        Fitting Function for T1
        """
        return a * np.exp(- t / T1) + b

    def _guess_init_params(self):

        a0 = self.signal[0] - self.signal[-1]
        b0 = self.signal[-1]
        
        mid_idx = np.argmin(np.abs(self.signal - (a0 / 2 + b0)))
        
        T10 = ((self.time[0] - self.time[mid_idx]) /
               np.log(1 - (self.signal[0] - self.signal[mid_idx]) / a0))

        self.p0 = [T10, a0, b0]
        # lb = [-np.inf, 0, np.min(self.signal)]
        # ub = [np.inf, (np.max(self.signal) - np.min(self.signal)),
        #       np.max(self.signal)]

    def _save_fit_results(self, popt, pcov):
        super()._save_fit_results(popt, pcov)

        self.T1 = popt[0]
        self.T1_sigma_err = np.sqrt(pcov[0, 0])

    def plot_result(self, fit_n_pts=1000):
        super().plot_result()

        # get most of the plotting done
        fig = self._plot_base()
        

        time_fit = np.linspace(self.time[0], self.time[-1], fit_n_pts)
        
        plt.plot(time_fit / self.time_scaler,
                 self.fit_func(time_fit, *(self.p0)),
                 label="Fit (Init. Param.)", lw=2, ls='--', color="orange")
        plt.plot(time_fit / self.time_scaler,
                 self.fit_func(time_fit, *(self.popt)),
                 label="Fit (Opt. Param.)", lw=2, color="red")
        plt.title(r"$T_1$ = %.5f $\pm$ %.5f " %
                  (self.T1 / self.time_scaler,
                   2 * self.T1_sigma_err / self.time_scaler) +
                  self.time_prefix + 's')
        plt.legend(fontsize='x-small')
        fig.tight_layout()
        plt.show()

class Ramsey(TimeDomain):
    """
    Class to perform analysis and visualize Ramsey fringes data.
    """
    def __init__(self, time, signal):
        super().__init__(time, signal)
        self.T2Ramsey = None
        self.delta_freq = None

    def fit_func(self, t, *args):
        """
        Fitting Function for Ramsey Fringes
            f(t) = a exp(-t/T2) cos[2π∆f(t-t0)] + b
        """
        
        if not self.fit_gaussian:
            T2, df, t0, a, b = args
            return a * np.exp(- t / T2) * np.cos(2 * np.pi * df * (t - t0)) + b
        else:
            Texp, Tgauss, df, t0, a, b = args
            return a * np.exp(- t / Texp - (t / Tgauss) ** 2) * np.cos(2 * np.pi * df * (t - t0)) + b

    def analyze(self, p0=None, plot=True, fit_gaussian=False, **kwargs):
        """
        Analyze the data with initial parameter `p0`.
        """
        self.fit_gaussian = fit_gaussian
        # set initial fit parameters
        self._set_init_params(p0)
        # perform fitting
        if self.lb is not None and self.ub is not None:
            popt, pcov = curve_fit(self.fit_func, self.time, self.signal,
                                   p0=self.p0, bounds=(self.lb, self.ub),
                                   **kwargs)
        else:
            popt, pcov = curve_fit(self.fit_func, self.time, self.signal,
                                   p0=self.p0, **kwargs)
        self.is_analyzed = True

        # save fit results
        self._save_fit_results(popt, pcov)

        if plot:
            self.plot_result()

    def _guess_init_params(self):
        b0 = np.mean(self.signal)
        signal0 = self.signal - b0
        amax = np.max(np.abs(signal0))

        # perform fft to find frequency of Ramsey fringes
        freq = np.fft.rfftfreq(len(self.signal),
                               d=(self.time[1] - self.time[0]))
        δf0 = freq[np.argmax(np.abs(np.fft.rfft(self.signal - np.mean(self.signal))))]
        df = freq[1] - freq[0] # frequency step of FFT

        # in-phase and quadrature envelope
        envComplex = _get_envelope(signal0, self.time, δf0)
        t00 = - np.angle(np.sum(envComplex)) / (2 * np.pi * δf0)

        env = np.abs(envComplex)

        if env[-1] < env[0]: # sanity check: make sure the envelope is decreasing over time
            # try:
            # mid_idx = np.argmin(np.abs(env - 0.5 * (env[-1] + env[0])))
            T20 = - (self.time[-1] - self.time[0]) / np.log(env[-1] / env[0])
        else:
            T20 = self.time[-1] - self.time[0]

        if not self.fit_gaussian:
            self.p0 = [T20, δf0, t00, amax, b0]
        else:
            self.p0 = [2 * T20, T20, δf0, t00, amax, b0]

    def _save_fit_results(self, popt, pcov):
        super()._save_fit_results(popt, pcov)

        if not self.fit_gaussian:
            self.T2Ramsey = popt[0]
            self.T2Ramsey_sigma_err = np.sqrt(pcov[0, 0])
            self.delta_freq = popt[1]
            self.delta_freq_sigma_err = np.sqrt(pcov[1, 1])
        else:
            self.Texp = popt[0]
            self.Texp_sigma_err = np.sqrt(pcov[0, 0])
            self.Tgauss = popt[1]
            self.Tgauss_sigma_err = np.sqrt(pcov[1, 1])
            self.delta_freq = popt[2]
            self.delta_freq_sigma_err = np.sqrt(pcov[2, 2])

    def plot_result(self, fit_n_pts=1000):
        super().plot_result()

        # get most of the plotting done
        fig = self._plot_base()

        time_fit = np.linspace(self.time[0], self.time[-1], fit_n_pts)

        plt.plot(time_fit / self.time_scaler,
                 self.fit_func(time_fit, *(self.p0)),
                 label="Fit (Init. Param.)", lw=2, ls='--', color="orange")
        plt.plot(time_fit / self.time_scaler,
                 self.fit_func(time_fit, *(self.popt)),
                 label="Fit (Opt. Param.)", lw=2, color="red")

        if not self.fit_gaussian:
            _, T2_prefix = number_with_si_prefix(self.T2Ramsey)
            T2_scaler = si_prefix_to_scaler(T2_prefix)
            
            T2_string = (r"$T_2^*$ = %.3f $\pm$ %.3f " %
                         (self.T2Ramsey / T2_scaler,
                          2 * self.T2Ramsey_sigma_err / T2_scaler) +
                          T2_prefix + 's')
        else:
            _, Texp_prefix = number_with_si_prefix(self.Texp)
            Texp_scaler = si_prefix_to_scaler(Texp_prefix)
            
            Texp_string = (r"$T_\mathrm{exp}$ = %.3f $\pm$ %.3f " %
                         (self.Texp / Texp_scaler,
                          2 * self.Texp_sigma_err / Texp_scaler) +
                          Texp_prefix + 's')
            
            _, Tgauss_prefix = number_with_si_prefix(self.Tgauss)
            Tgauss_scaler = si_prefix_to_scaler(Tgauss_prefix)
            
            Tgauss_string = (r"$T_\mathrm{gauss}$ = %.3f $\pm$ %.3f " %
                         (self.Tgauss / Tgauss_scaler,
                          2 * self.Tgauss_sigma_err / Tgauss_scaler) +
                          Tgauss_prefix + 's')
            
            T2_string = ', '.join([Texp_string, Tgauss_string])

        _, delta_freq_prefix = number_with_si_prefix(self.delta_freq)
        delta_freq_scaler = si_prefix_to_scaler(delta_freq_prefix)

        delta_freq_string = (r"$\Delta f$ = %.3f $\pm$ %.3f " %
                     (self.delta_freq / delta_freq_scaler,
                      2 * self.delta_freq_sigma_err / delta_freq_scaler) +
                     delta_freq_prefix + 'Hz')

        plt.title(', '.join([T2_string, delta_freq_string]), fontsize='small')
        plt.legend(loc=0, fontsize='x-small')
        fig.tight_layout()
        plt.show()
        return fig

class RamseyWithGaussian(TimeDomain):
    """
    Class to perform analysis and visualize Ramsey fringes data with Gaussian envelopes
    XZ: added an optional p0 to pass in initial parameters if the automatic fitting fails. 20201230
    """
    def __init__(self, time, signal, p0 = None):
        super().__init__(time, signal)
        self.T_phi1 = None
        self.T_phi2 = None
        self.delta_freq = None
        if p0 is not None:
            self.p0 = p0

    def fit_func(self, t, T_phi1, T_phi2, δf, t0, a, b, c, Td):
        """
        Fitting Function for Ramsey Fringes with Gaussian Envelope
        """
        return (a * np.exp(- t / T_phi1 - (t / T_phi2) ** 2) * np.cos(2 * np.pi * δf * (t - t0)) + b +
                c * np.exp(-t / Td))

    def _guess_init_params(self):
        if self.p0 == None:
            b0 = np.mean(self.signal)
            signal0 = self.signal - b0
            a0 = np.max(np.abs(signal0))
    
            # perform fft to find frequency of Ramsey fringes
            freq = np.fft.rfftfreq(len(self.signal),
                                   d=(self.time[1]-self.time[0]))
            δf0 = freq[np.argmax(np.abs(np.fft.rfft(self.signal - np.mean(self.signal))))]
            T_phi10 = (self.time[-1] - self.time[0]) / 3
            T_phi20 = (self.time[-1] - self.time[0]) / 3
            self.p0 = [T_phi10, T_phi20, δf0, 0.0, a0, b0, 0.0, self.time[-1] - self.time[0]]

    def _save_fit_results(self, popt, pcov):
        super()._save_fit_results(popt, pcov)

        self.T_phi1 = popt[0]
        self.T_phi2 = popt[1]
        self.delta_freq = popt[2]

    def plot_result(self, fit_n_pts=1000):
        super().plot_result()

        # get most of the plotting done
        fig = self._plot_base()

        time_fit = np.linspace(self.time[0], self.time[-1], fit_n_pts)

        plt.plot(time_fit / 1e-6, self.fit_func(time_fit, *(self.popt)),
                 label="Fit", lw=2, color="red")
        plt.title(r"$T_{\phi,1} = %.5f \mu\mathrm{s}$, $T_{\phi,2} = %.5f \mu\mathrm{s}$" % (self.T_phi1 / 1e-6,
                                                                                             self.T_phi2 / 1e-6))
        fig.tight_layout()
        plt.show()

# TODO: RamseyWithBeating(TimeDomain):

class HahnEcho(TimeDomain):
    """
    Class to analyze and visualize Hahn echo decay (T2 Hahn echo experiment) data.
    """
    def __init__(self, time, signal):
        super().__init__(time, signal)
        self.T2Echo = None

    def fit_func(self, t, *args):
        """
        Fitting Function for Hahn Echo
        """
        if not self.fit_gaussian:
            T2, a, b = args
            return a * np.exp(- t / T2) + b
        else:
            Texp, Tgauss, a, b = args
            return a * np.exp(- t / Texp - (t / Tgauss) ** 2) + b

    def analyze(self, p0=None, plot=True, fit_gaussian=False, **kwargs):
        """
        Analyze the data with initial parameter `p0`.
        """
        self.fit_gaussian = fit_gaussian
        # set initial fit parameters
        self._set_init_params(p0)
        # perform fitting
        if self.lb is not None and self.ub is not None:
            popt, pcov = curve_fit(self.fit_func, self.time, self.signal,
                                   p0=self.p0, bounds=(self.lb, self.ub),
                                   **kwargs)
        else:
            popt, pcov = curve_fit(self.fit_func, self.time, self.signal,
                                   p0=self.p0, **kwargs)
        self.is_analyzed = True

        # save fit results
        self._save_fit_results(popt, pcov)

        if plot:
            self.plot_result()

    def _guess_init_params(self):
        a0 = self.signal[0] - self.signal[-1]
        b0 = self.signal[-1]

        mid_idx = np.argmin(np.abs(self.signal - (a0 / 2 + b0)))
        T20 = ((self.time[0] - self.time[mid_idx]) /
               np.log(1 - (self.signal[0] - self.signal[mid_idx]) / a0))

        if not self.fit_gaussian:
            self.p0 = [T20, a0, b0]
        else:
            N_linear = 10
            slope, _ = np.polyfit(self.time[:N_linear], self.signal[:N_linear], 1)
            Texp0 = 1 / np.abs(slope / a0)
            
            t_mid = self.time[mid_idx]
            Tgauss0 = t_mid / np.sqrt(np.log(2) - t_mid / Texp0)           
            
            self.p0 = [Texp0, Tgauss0, a0, b0]

            self.lb = [0.5 * Texp0, 0.5 * Tgauss0, -np.inf, -np.inf]
            self.ub = [1.5 * Texp0, 1.5 * Tgauss0, np.inf, np.inf]


    def _save_fit_results(self, popt, pcov):
        super()._save_fit_results(popt, pcov)
        if not self.fit_gaussian:
            self.T2Echo = popt[0]
            self.T2Echo_sigma_err = np.sqrt(pcov[0, 0])
        else:
            self.Texp = popt[0]
            self.Texp_sigma_err = np.sqrt(pcov[0, 0])
            self.Tgauss = popt[1]
            self.Tgauss_sigma_err = np.sqrt(pcov[1, 1])
            
    def plot_result(self, fit_n_pts=1000):
        super().plot_result()

        # get most of the plotting done
        fig = self._plot_base()

        time_fit = np.linspace(self.time[0], self.time[-1], fit_n_pts)

        plt.plot(time_fit / self.time_scaler,
                 self.fit_func(time_fit, *(self.p0)),
                 label="Fit (Init. Param.)", ls='--', lw=2, color="orange")
        plt.plot(time_fit / self.time_scaler,
                 self.fit_func(time_fit, *(self.popt)),
                 label="Fit (Opt. Param.)", lw=2, color="red")

        if not self.fit_gaussian:
            _, T2Echo_prefix = number_with_si_prefix(self.T2Echo)
            T2Echo_scaler = si_prefix_to_scaler(T2Echo_prefix)
            
            T2_string = (r"$T_{2E}$ = %.3f $\pm$ %.3f " % (self.T2Echo / T2Echo_scaler,
                                                           2 * self.T2Echo_sigma_err / T2Echo_scaler) +
                         T2Echo_prefix + 's')
        else:
            _, Texp_prefix = number_with_si_prefix(self.Texp)
            Texp_scaler = si_prefix_to_scaler(Texp_prefix)

            _, Tgauss_prefix = number_with_si_prefix(self.Tgauss)
            Tgauss_scaler = si_prefix_to_scaler(Tgauss_prefix)

            Texp_string = (r"$T_\mathrm{exp}$ = %.3f $\pm$ %.3f " % (self.Texp / Texp_scaler,
                                                                     2 * self.Texp_sigma_err / Texp_scaler) +
                           Texp_prefix + 's')
            Tgauss_string = (r"$T_\mathrm{gauss}$ = %.3f $\pm$ %.3f " % (self.Tgauss / Tgauss_scaler,
                                                                         2 * self.Tgauss_sigma_err / Tgauss_scaler) +
                             Tgauss_prefix + 's')
            T2_string = ', '.join([Texp_string, Tgauss_string])
        plt.title(T2_string, fontsize='small')
        fig.tight_layout()
        plt.show()

class DRAGMotzoiXY(TimeDomain):
    '''
    Class to analyze and visualize DRAG pulse calibration experiment
    for determining Motzoi parameter (beta).
    '''
    def __init__(self, beta, signal, labels=None):
        # initialize parameters
        self.beta = beta
        self.signal = signal
        self.n_seq, self.n_pts = self.signal.shape
        self.is_analyzed = False
        if labels is None:
            self.sequence = [str(i) for i in range(self.n_seq0)]
        else:
            self.sequence = labels

        self.p0 = None
        self.lb = None
        self.ub = None

        self.popt = None
        self.pcov = None

        self.beta0 = None
        self.signal0 = None
        
    def fit_func(self, beta, beta0, signal0, *a):
        """
        Fitting Function for DRAG Motzoi XY experiment. The function returns
        an array of length n_seq * n_pts.
        """
        
        N = len(beta) // self.n_seq
        
        return np.hstack(
            [(a[i] * (beta[i * N:((i + 1) * N)] - beta0) +
              signal0) for i in range(self.n_seq)])

    def _guess_init_params(self):
        
        a0 = [((self.signal[i, -1] - self.signal[i, 0]) /
               (self.beta[-1] - self.beta[0])) for i in range(self.n_seq)]

        closest_idx = np.argmin(np.var(self.signal, axis=0))
        beta0 = self.beta[closest_idx]
        signal0 = np.mean(self.signal[:, closest_idx])

        self.p0 = [beta0, signal0, *a0]

    def analyze(self, p0=None, plot=True, **kwargs):
        """
        Analyze the data with initial parameter `p0`.
        """
        # set initial fit parameters
        self._set_init_params(p0)
        # perform fitting
        if self.lb is not None and self.ub is not None:
            popt, pcov = curve_fit(self.fit_func,
                                   np.hstack([self.beta] * self.n_seq),
                                   self.signal.flatten(),
                                   p0=self.p0, bounds=(self.lb, self.ub),
                                   **kwargs)
        else:
            popt, pcov = curve_fit(self.fit_func,
                                   np.hstack([self.beta] * self.n_seq),
                                   self.signal.flatten(),
                                   p0=self.p0, **kwargs)
        self.is_analyzed = True

        # save fit results
        self._save_fit_results(popt, pcov)

        if plot:
            self.plot_result()

    def _save_fit_results(self, popt, pcov):
        super()._save_fit_results(popt, pcov)

        self.beta_opt = popt[0]
        self.signal_opt = popt[1]
        self.beta_opt_sigma_err = np.sqrt(pcov[0, 0])
        self.signal_opt_sigma_err = np.sqrt(pcov[1, 1])

    def plot_result(self, fit_n_pts=1000):
        super().plot_result()

        fig = plt.figure()

        # plot data
        for i in range(self.n_seq):
            plt.plot(self.beta, self.signal[i, :], '.',
                     label="Data " + self.sequence[i], color="C%d" % i)
        plt.xlabel(r"DRAG param. $\beta$")
        plt.ylabel("Signal")

        beta_fit = np.linspace(self.beta[0], self.beta[-1], fit_n_pts)

        fits0 = self.fit_func(np.hstack([beta_fit] * self.n_seq),
                              *(self.p0)).reshape(self.n_seq, fit_n_pts)
        fits = self.fit_func(np.hstack([beta_fit] * self.n_seq),
                             *(self.popt)).reshape(self.n_seq, fit_n_pts)
        
        for i in range(self.n_seq):
            plt.plot(beta_fit, fits0[i, :],
                     label="Fit " + self.sequence[i] + " (init. param.)", lw=1, ls='--',
                     color="C%d" % i, alpha=0.5)
        for i in range(self.n_seq):
            plt.plot(beta_fit, fits[i, :],
                     label="Fit " + self.sequence[i] + " (opt. param.)", lw=2, ls='-', color="C%d" % i)
            
        plt.axvline(x=self.beta_opt, ls='--', color='black')
        plt.axhline(y=self.signal_opt, ls='--', color='black')

        plt.title(r"$\beta_\mathrm{opt} = %.3f \pm %.3f$" %
                  (self.beta_opt, 2 * self.beta_opt_sigma_err))
        plt.legend(loc='lower left', fontsize='x-small', ncol=self.n_seq)
        fig.tight_layout()
        plt.show()


class AllXY(TimeDomain):
    '''
    Class to analyze and visualize AllXY experiment result.
    '''

    def __init__(self, sequence, signal):
        self.sequence = sequence
        self.signal = signal
        self.n_seq = len(self.sequence)
        self.seq_index = np.arange(self.n_seq)        
        seq_rot_map = {"X": (0.0, 1.0), "x": (0.0, 0.5), "Y": (0.25, 1.0),
                       "y": (0.25, 0.5), "I": (0.0, 0.0)}
        
        def rot(angle, amp):
            theta = amp * np.pi
            if angle == 0.0: # rotation along x axis
                return np.array([[1, 0, 0],
                                 [0, np.cos(theta), -np.sin(theta)],
                                 [0, np.sin(theta), np.cos(theta)]])
            if angle == 0.25: # rotation along y axis
                return np.array([[np.cos(theta), 0, np.sin(theta)],
                                 [0, 1, 0],
                                 [-np.sin(theta), 0, np.cos(theta)]])
        
        def pop(seq):          
            state = np.array([0, 0, -1])
            
            for gate in seq:
                state = np.matmul(rot(*seq_rot_map[gate]), state)
            
            return (state[-1] + 1) / 2

        self.sequence_pop = np.array([pop(seq) for seq in self.sequence])

        self.p0 = None
        self.lb = None
        self.ub = None

        self.popt = None
        self.pcov = None

    def fit_func(self, n_vec, a, b):
        '''
        Fitting function for AllXY experiment.
        '''

        seq_pop = np.array(
            [self.sequence_pop[int(np.round(n))] for n in n_vec])
        return a * seq_pop + b

    def _guess_init_params(self):
        
        high0 = np.mean(self.signal[self.sequence_pop == 1.0])
        mid0 = np.mean(self.signal[self.sequence_pop == 0.5])
        low0 = np.mean(self.signal[self.sequence_pop == 0.0])

        mid = np.mean([high0, mid0, low0])

        b0 = low0
        a0 = 2 * (mid - b0)
        
        self.p0 = [a0, b0]

    def analyze(self, p0=None, plot=True, **kwargs):
        """
        Analyze the data with initial parameter `p0`.
        """
        # set initial fit parameters
        self._set_init_params(p0)
        # perform fitting
        if self.lb is not None and self.ub is not None:
            popt, pcov = curve_fit(self.fit_func,
                                   self.seq_index,
                                   self.signal,
                                   p0=self.p0, bounds=(self.lb, self.ub),
                                   **kwargs)
        else:
            popt, pcov = curve_fit(self.fit_func,
                                   self.seq_index,
                                   self.signal,
                                   p0=self.p0, **kwargs)
        self.is_analyzed = True

        # save fit results
        self._save_fit_results(popt, pcov)

        if plot:
            self.plot_result()

    def _save_fit_results(self, popt, pcov):
        super()._save_fit_results(popt, pcov)
        a, b = self.popt
        self.error_AllXY = np.sum(
            np.abs(self.sequence_pop - (self.signal - b) / a))

    def plot_result(self, fit_n_pts=1000):
        super().plot_result()

        fig = plt.figure()

        # plot data
        plt.plot(self.seq_index, self.signal, '.', color='black',
                 label='Data')
        plt.xticks(ticks=self.seq_index,
                   labels=[''.join(seq) for seq in self.sequence])
        plt.xlabel("Sequence")
        plt.ylabel("Signal")

        n_fit = np.linspace(self.seq_index[0], self.seq_index[-1], fit_n_pts)
        
        plt.plot(n_fit, self.fit_func(n_fit, *(self.p0)),
                 label="Fit (init. param.)", lw=2, ls='--',
                 color="orange")
        plt.plot(n_fit, self.fit_func(n_fit, *(self.popt)),
                 label="Fit (opt. param.)", lw=2, ls='-',
                 color="red")
    
        plt.title(r"Normalized AllXY error: $\mathcal{E}_\mathrm{AllXY}$ = %.3f" % self.error_AllXY)
        plt.legend(loc='upper left', fontsize='x-small')
        fig.tight_layout()
        plt.show()

class PulseTrain(TimeDomain):
    def __init__(self, correction, repetition, signal):
        # initialize parameters
        self.correction = correction
        self.repetition = repetition
        self.signal = signal
        self.n_correction, self.n_pts = self.signal.shape

        self.is_analyzed = False

        self.p0 = None
        self.popt = None
        self.pcov = None

    def fit_func(self, repetition, eps0, N1, *args):
        """
        Fitting Function for Pulse Train experiment.
        """
        
        N = len(repetition) // self.n_correction
        
        A = args[:self.n_correction]
        B = args[self.n_correction:]

        decay = [np.exp(-repetition[(i * N):((i + 1) * N)] / N1)
                 for i in range(self.n_correction)]
        oscillation = [np.cos(np.pi * (1 + eps0) *
                              (1 + self.correction[i]) *
                              (2 * repetition[(i * N):((i + 1) * N)] + 0.5))
                       for i in range(self.n_correction)]
        return np.hstack([A[i] * decay[i] * oscillation[i] + B[i]
                          for i in range(self.n_correction)])

    def _guess_init_params(self):
        B0 = np.mean(self.signal)        
        mean_zero_signal = self.signal - B0
        
        idx = np.argmax(np.abs(mean_zero_signal.flatten()))
        
        a0 = np.abs(mean_zero_signal.flatten())[idx]
        row, col = idx // self.n_pts, idx % self.n_pts
        a1 = np.max(np.abs(mean_zero_signal)[:, -1])
        N1 = - (self.repetition[-1] - self.repetition[col]) / np.log(a1 / a0)

        A0 = - a0 * np.exp(self.repetition[col] / N1)
        
        zero_idx = np.argmin(np.var(self.signal, axis=-1))
        
        self.p0 = [-self.correction[zero_idx],
                   N1, *([A0] * self.n_correction), *([B0] * self.n_correction)]

    def analyze(self, p0=None, plot=True, **kwargs):
        """
        Analyze the data with initial parameter `p0`.
        """
        # set initial fit parameters
        self._set_init_params(p0)
        # perform fitting
        popt, pcov = curve_fit(self.fit_func,
                               np.hstack([self.repetition] * self.n_correction),
                               self.signal.flatten(),
                               p0=self.p0, **kwargs)
        self.is_analyzed = True

        # save fit results
        self._save_fit_results(popt, pcov)

        if plot:
            self.plot_result()

    def _save_fit_results(self, popt, pcov):
        super()._save_fit_results(popt, pcov)

        self.eps0 = popt[0]
        self.opt_correction = 1 / (1 + self.eps0) - 1
        self.N1 = popt[1]

        self.eps0_sigma_err = np.sqrt(pcov[0, 0])
        self.N1_sigma_err = np.sqrt(pcov[1, 1])
        self.opt_correction_sigma_err = (1 + self.opt_correction) ** 2 * self.eps0_sigma_err

    def plot_result(self):
        super().plot_result()

        fig = plt.figure()

        plt.subplot(1, 2, 1)
        plt.pcolormesh(self.repetition, self.correction * 100, self.signal,
                       shading='nearest', cmap=plt.get_cmap('jet'))
        plt.axhline(self.opt_correction * 100, ls='--', color='black')
        plt.xlabel('Number of Repetitions')
        plt.ylabel(r'Amplitude Correction (%)')
        cb = plt.colorbar()
        cb.ax.set_title(r'Signal', fontsize='x-small')

        n_fit = self.repetition
        fit_ = self.fit_func(np.hstack([n_fit] * self.n_correction),
                             *self.popt).reshape(self.n_correction, -1)
        plt.subplot(1, 2, 2)
        for k, acorr in enumerate(self.correction):
            plt.plot(self.repetition, self.signal[k, :], '.', color='C%d' % k,
                     label='Data (%.2f' % acorr + r' %)', ms=3)
            plt.plot(n_fit, fit_[k, :], '-', color='C%d' % k,
                     label='Fit (%.2f' % acorr + r' %)')
        plt.xlabel('Number of Repetitions')
        plt.ylabel('Signal')
        # plt.legend(fontsize=4)

        plt.suptitle('Opt. Amp. Correction: $%.3f \pm %.3f$' %
                     (self.opt_correction * 100,
                      2 * self.opt_correction_sigma_err * 100) + r' %')
        plt.tight_layout()
        plt.show()


class RandomizedBenchmarking(TimeDomain):
    def __init__(self, n_clifford, signal, interleaved_signals=None,
                 interleaved_gates=None, n_qubit=1):

        self.n_clifford = n_clifford
        self.signal = signal
        self.n_sequence, self.n_max_clifford = signal.shape
        self.mean_signal = np.mean(self.signal, axis=0)
        
        self.interleaved_signals = interleaved_signals
        if self.interleaved_signals is not None:
            self.n_interleaved_gates = len(self.interleaved_signals)
            self.mean_interleaved_signals = [np.mean(sig, axis=0) for sig in self.interleaved_signals]
            self.interleaved_gates = interleaved_gates
            if self.interleaved_gates is None:
                self.interleaved_gates = ['gate%d' % i for i in range(self.n_interleaved_gates)]

        self.n_qubit = n_qubit

        self.p0_list = []
        self.popt_list = []
        self.pcov_list = []

    def fit_func(self, m, p, A, B):
        '''
        Fitting function for Randomized Benchmarking experiment.
        '''
        return A * p ** m + B

    def _guess_init_params(self, mean_signal):
        A0 = mean_signal[0] - mean_signal[-1]
        B0 = mean_signal[-1]

        mid_idx = np.argmin(np.abs(mean_signal - (A0 / 2 + B0)))
        M1 = ((self.n_clifford[0] - self.n_clifford[mid_idx]) /
               np.log(1 - (mean_signal[0] - mean_signal[mid_idx]) / A0))
        p0 = np.exp(-1 / M1)
        self.p0_list.append([p0, A0, B0])

    def analyze(self, plot=True, **kwargs):
        """
        Analyze the data with initial parameter `p0`.
        """
        # fitting of RB data
        self._guess_init_params(self.mean_signal)
        popt, pcov = curve_fit(self.fit_func, self.n_clifford,
                               self.mean_signal, p0=self.p0_list[0], **kwargs)
        self.popt_list.append(popt)
        self.pcov_list.append(pcov)
        
        # fitting of interleaved RB data
        if self.interleaved_signals is not None:
            for idx in range(self.n_interleaved_gates):
                self._guess_init_params(self.mean_interleaved_signals[idx])
                popt, pcov = curve_fit(self.fit_func, self.n_clifford,
                                       self.mean_interleaved_signals[idx],
                                       p0=self.p0_list[1 + idx], **kwargs)
                self.popt_list.append(popt)
                self.pcov_list.append(pcov)              
        
        self.is_analyzed = True

        # depolarizing parameter list
        self.p_list = [popt[0] for popt in self.popt_list]
        self.p_sigma_err_list = [np.sqrt(pcov[0, 0]) for pcov in self.pcov_list]
        
        # clifford gate set fidelity
        self.r_clifford = (1 - self.p_list[0]) * (1 - 1 / 2 ** self.n_qubit)
        self.r_clifford_sigma_err = self.p_sigma_err_list[0] * (1 - 1 / 2 ** self.n_qubit)
        self.fidelity_clifford = 1 - self.r_clifford
        self.fidelity_clifford_sigma_err = self.r_clifford_sigma_err

        # target gate fidelity from IRB
        self.r_gate = []
        self.r_gate_sigma_err = []
        self.fidelity_gate = []
        self.fidelity_gate_sigma_err = []

        if self.interleaved_signals is not None:
            for gate_idx in range(1, self.n_interleaved_gates + 1):
                r_gate = (1 - self.p_list[gate_idx] / self.p_list[0]) * (1 - 1 / 2 ** self.n_qubit)
                r_gate_sigma_err = ((self.p_list[gate_idx] / self.p_list[0]) * (1 - 1 / 2 ** self.n_qubit) *
                                    np.sqrt((self.p_sigma_err_list[gate_idx] / self.p_list[gate_idx]) ** 2 +
                                            (self.p_sigma_err_list[0] / self.p_list[0]) ** 2))
                self.r_gate.append(r_gate)
                self.r_gate_sigma_err.append(r_gate_sigma_err)
                self.fidelity_gate.append(1 - r_gate)
                self.fidelity_gate_sigma_err.append(r_gate_sigma_err)
        
        if plot:
            self.plot_result()

    def plot_result(self, fit_n_pts=1000):
        fig = plt.figure()

        # plot data
        for i in range(self.n_sequence):
            plt.plot(self.n_clifford, self.signal[i, :], '.', color='C0',
                     alpha=0.1, ms=2)
        plt.plot(self.n_clifford, self.mean_signal, '.', color='C0',
                 label='Avg. Data (Clifford)', markeredgecolor='black',
                 markeredgewidth=0.8, ms=8)
        
        if self.interleaved_signals is not None:
            for k in range(self.n_interleaved_gates):
                for i in range(self.n_sequence):
                    plt.plot(self.n_clifford, self.interleaved_signals[k][i, :],
                             '.', color='C%d' % (k + 1), alpha=0.1, ms=2)
                plt.plot(self.n_clifford, self.mean_interleaved_signals[k],
                         '.', color='C%d' % (k + 1), markeredgecolor='black',
                         markeredgewidth=0.8, ms=8,
                         label='Avg. Data (Interleaved ' +  self.interleaved_gates[k] + ')')
        
        plt.xlabel("Number of Cliffords")
        plt.ylabel("Signal")

        n_clifford_fit = np.linspace(self.n_clifford[0], self.n_clifford[-1], fit_n_pts)
        
        plt.plot(n_clifford_fit, self.fit_func(n_clifford_fit, *(self.popt_list[0])),
                 label=("Fit (Clifford): " +
                        r'$r_\mathrm{Clifford}=%.3f \pm %.3f $' % (self.r_clifford * 100,
                                                                   2 * self.r_clifford_sigma_err * 100) + '%'),
                 lw=2, ls='-', color="C0")

        
        if self.interleaved_signals is not None:
            for k in range(self.n_interleaved_gates):
                plt.plot(n_clifford_fit, self.fit_func(n_clifford_fit, *(self.popt_list[k + 1])),
                         label=("Fit (Interleaved " + self.interleaved_gates[k] + '): ' +
                                r'$r_\mathrm{%s}=%.3f \pm %.3f $' %
                                (self.interleaved_gates[k], self.r_gate[k] * 100, 2 * self.r_gate_sigma_err[k] * 100) + '%'),
                         lw=2, ls='-', color="C%d" % (k + 1))

        plt.title('Randomized Benchmarking (%d random sequences)' % self.n_sequence)
        plt.legend(loc='best', fontsize='x-small')
        plt.tight_layout()
        plt.show()

class EasyReadout:
    """
    Class to implement easy analysis of readout data.
    """
    def __init__(self, data, blob_locations=None,
                 readout_type='phase', ang_tol=np.pi/1000):
        self.data = data
        self.n_pts = len(self.data)

        if blob_locations is None:
            # This will be overwritten in `self._project_to_line` function call
            self.n = 1.0 + 0.0j     # unit vector connecting the ground and excited state points in the complex plane
            self.v_g = 0.0 + 0.0j   # ground state point in the complex plane
            self.v_e = 0.0 + 0.0j   # excited state point in the complex plane
    
            # qubit population extracted by fitting to a line and
            self.population = None
            self._project_to_line(readout_type, ang_tol)
        else:
            self.v_g = blob_locations[0]
            self.v_e = blob_locations[1]
            
            self.population = np.real((data - self.v_g) / (self.v_e - self.v_g))

    def _project_to_line(self, readout_type, ang_tol):
        """
        Fit a straight line to a full complex dataset of qubit readout
        """
        # fit a straight line on the full data (A x + B y + C = 0)
        A, B, C = self._fit_line(ang_tol)

        # find (approximate) ground state and excited state voltage
        if readout_type == 'magnitude':
            mag_data = np.abs(self.data)
            self.v_g = self._find_point_closest(self.data[np.argmin(mag_data)], A, B, C)
            self.v_e = self._find_point_closest(self.data[np.argmax(mag_data)], A, B, C)
        elif readout_type == 'phase':
            phase_data = np.unwrap(np.angle(self.data))
            self.v_g = self._find_point_closest(self.data[np.argmin(phase_data)], A, B, C)
            self.v_e = self._find_point_closest(self.data[np.argmax(phase_data)], A, B, C)

        # unit vector along the fitted line
        self.n = (self.v_e - self.v_g) / np.abs(self.v_e - self.v_g)

        # projection of complex data to a line v_orig + x * n
        self.population = (self._inner_product(self.data - self.v_g, self.n) /
                           np.abs(self.v_e - self.v_g))

    def _inner_product(self, z1, z2):
        """
        Element-wise inner product between complex vectors or between a complex vector and a complex number.
        """
        return z1.real * z2.real + z1.imag * z2.imag

    def _fit_line(self, ang_tol):
        """
        Fit a straight line to a complex dataset such that standard deviation
        of data projected onto the line is minimized. `N` determines the
        precision in the angle of the new axis.
        """
        v = self.data

        N = int(np.ceil(np.pi / ang_tol))
        theta = np.linspace(0, np.pi, N)

        std, mean = np.zeros(N), np.zeros(N)
        for m, theta_ in enumerate(theta):
            # projection of data points to axis rotated by θ from the real axis
            x = self._inner_product(v, np.cos(theta_) + 1j * np.sin(theta_))
            std[m] = np.std(x)
            mean[m] = np.mean(x)

        # find the angle with minimum standard deviation of data
        m_min = np.argmin(std)
        theta_min, mean_min = theta[m_min], mean[m_min]

        v0 = mean_min * (np.cos(theta_min) + 1j * np.sin(theta_min))

        A, B, C = np.cos(theta_min), np.sin(theta_min), - mean_min

        return A, B, C

    def _find_point_closest(self, v, A, B, C):
        """
        Find a point in line y = ax + b closest to the point v = x + 1j* y.
        This performs a projection of the measured voltage onto a line connecting
        that of state 0 and state 1.
        """
        # y = ax + b <==> Ax + By + C = 0 (A = a, B = -1, C = b)
        x, y = v.real, v.imag
        xx = (B * (B * x - A * y) - A * C) / (A ** 2 + B ** 2)
        yy = (A * (-B * x + A * y) - B * C) / (A ** 2 + B ** 2)
        return xx + 1j * yy

class MeasInducedDephasing:
    _bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
    def __init__(self, xi, phi, signal, tau_p):
        self.signal = signal
        self.amplitude = xi
        self.pulse_duration = tau_p
        self.phase = phi
        self.dephasing_rate = None

        self.ramsey_fits = []
        envelopes = []
        envelope_errs = []

        for idx in range(len(xi)):
            r = RamseyWithVirtualZRotation(phi, self.signal[idx, :])
            r.analyze(plot=False, maxfev=10000)
            self.ramsey_fits.append(r) 
            envelopes.append(r.amplitude)
            envelope_errs.append(r.amplitude_err)
        self.ramsey_envelope = np.array(np.abs(envelopes))
        self.ramsey_envelope_error = np.array(envelope_errs)

    def _guess_init_params(self):
        
        c = self.ramsey_envelope
        xi = self.amplitude
        tau_p = self.pulse_duration

        gamma_m = - np.log(c[0] / c[-1]) / (tau_p * (xi[0] ** 2 - xi[-1] ** 2))
        c0 = c[0] * np.exp(gamma_m * tau_p * xi[0] ** 2)
        
        self.p0 = np.array([gamma_m, c0])

    def fit_func(self, xi, gamma_m, c0):
        tau_p = self.pulse_duration
        return c0 * np.exp(-gamma_m * tau_p * xi ** 2)
    
    def analyze(self, plot=True):
        
        self._guess_init_params()
        
        self.popt, self.pcov = curve_fit(self.fit_func, self.amplitude,
                                         self.ramsey_envelope, p0=self.p0)
        self.dephasing_rate = self.popt[0]
        self.dephasing_rate_err = np.sqrt(self.pcov[0, 0])
        if plot:
            self.plot_result()
    
    def plot_result(self):

        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(2, 1)

        n_ramsey_plot = 4
        sgs0 = gs[0].subgridspec(1, n_ramsey_plot)
        sgs1 = gs[1].subgridspec(1, 2)
        ramsey_axes = [fig.add_subplot(sgs0[k]) for k in range(n_ramsey_plot)]
        fit_axes = [fig.add_subplot(sgs1[k]) for k in range(2)]

        n_amps = len(self.amplitude)
        plot_indices = [int(k * (n_amps - 1) /
                            (n_ramsey_plot - 1)) for k in range(n_ramsey_plot)]
        
        smin, smax = np.min(self.signal), np.max(self.signal)
        ds = smax - smin
        for idx, p in enumerate(plot_indices):
            xi = self.amplitude[p]
            r = self.ramsey_fits[p]
            amp, amp_err = r.amplitude, r.amplitude_err
            ramsey_axes[idx].plot(self.phase, r.signal, '.', color='black')
            ramsey_axes[idx].plot(self.phase, r.fit_func(self.phase, *r.popt),
                                  color="C%d" % idx,
                                  label=r"$%.3f\pm%.3f$" % (amp, 2 * amp_err))
            ramsey_axes[idx].set_title(r"$\xi = %.3f$" % xi, fontsize='small')
            ramsey_axes[idx].set_ylim([smin - 0.1 * ds, smax + 0.1 * ds])
            ramsey_axes[idx].legend(loc=1, fontsize='x-small')
        
        p = fit_axes[0].pcolormesh(self.phase, self.amplitude, self.signal, shading='nearest')
        fig.colorbar(p, ax=fit_axes[0])
        fit_axes[0].set_xlabel(r"Virtual rot. phase $\phi$ (rad)")
        fit_axes[0].set_ylabel(r"Relative RO amp. $\xi$")
        
        fit_axes[1].errorbar(self.amplitude, self.ramsey_envelope,
                             yerr=2 * self.ramsey_envelope_error, fmt='.', color='black')
        fit_axes[1].plot(self.amplitude, self.fit_func(self.amplitude, *self.popt),
                         label=r"$\bar{\Gamma}/2\pi=%.3f \pm %.3f$ MHz" % (self.dephasing_rate / (2 * np.pi * 1e6),
                                                                           2 * self.dephasing_rate_err / (2 * np.pi * 1e6)),
                         color='blue', lw=2)
        fit_axes[1].set_xlabel(r"Relative RO amp. $\xi$")
        fit_axes[1].set_ylabel(r"Ramsey envelope $c(\xi)$")

from qutip import destroy, qeye, tensor, basis, Options, mesolve

class VacuumRabiChevron(TimeDomain):
    def __init__(self, time, amp, signal, amp_polyorder: Optional[int]=1,
                 excited_qubit: Optional[str]='q1',
                 tuned_qubit: Optional[str]='q1',
                 measured_qubit: Optional[str]='q1'):
        self.signal = signal
        self.time = time
        self.amp = amp
        self.amp_polyorder = amp_polyorder
        
        if excited_qubit not in ['q1', 'q2']:
            raise ValueError(
                'The specified keyword argument `excited_qubit=%s`' % str(excited_qubit) +
                ' is not supported. Use either "q1" or "q2" instead.')        
        else:
            self.excited_qubit = excited_qubit

        if tuned_qubit not in ['q1', 'q2']:
            raise ValueError(
                'The specified keyword argument `tuned_qubit=%s`' % str(tuned_qubit) +
                ' is not supported. Use either "q1" or "q2" instead.')        
        else:
            self.tuned_qubit = tuned_qubit

        if measured_qubit not in ['q1', 'q2']:
            raise ValueError(
                'The specified keyword argument `measured_qubit=%s`' % str(measured_qubit) +
                ' is not supported. Use either "q1" or "q2" instead.')        
        else:
            self.measured_qubit = measured_qubit

        # qutip operators
        sm = destroy(2)
        sp = sm.dag()
        sz = 2 * sp * sm - qeye(2)

        self.qutip_ops = {
            'sm1': tensor(sm, qeye(2)), 'sp1': tensor(sp, qeye(2)),
            'sz1': tensor(sz, qeye(2)), 'sm2': tensor(qeye(2), sm),
            'sp2': tensor(qeye(2), sp), 'sz2': tensor(qeye(2), sz)
            }

    def fit_func(self, time, amp, params):
        '''
        

        Parameters
        ----------
        time : TYPE
            DESCRIPTION.
        amp : TYPE
            DESCRIPTION.
        params : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        # create an array to store the data
        result_arr = np.zeros((len(amp), len(time)))

        # unpack free params
        g, Gamma1_q1, Gamma1_q2, Gamma_phi_q1, Gamma_phi_q2, r0, r1, amp0, *c = params
    
        dz = amp - amp0
        Delta_list = np.sum([c[i] * dz ** (i + 1) for i in range(len(c))],
                            axis=0)

        # load qutip operators
        sm1, sp1, sz1 = self.qutip_ops['sm1'], self.qutip_ops['sp1'], self.qutip_ops['sz1']
        sm2, sp2, sz2 = self.qutip_ops['sm2'], self.qutip_ops['sp2'], self.qutip_ops['sz2']

        # interaction Hamiltonian
        Hint = g * (sm1 * sp2 + sp1 * sm2)
        # collapse operators (decay, pure dephasing)
        c_ops = [np.sqrt(Gamma1_q1) * sm1, np.sqrt(Gamma1_q2) * sm2,
                 np.sqrt(Gamma_phi_q1) * sz1, np.sqrt(Gamma_phi_q2) * sz2]

        if self.measured_qubit == 'q1':
            e_ops = [sp1 * sm1]
        elif self.measured_qubit == 'q2':
            e_ops = [sp2 * sm2]

        if self.excited_qubit == 'q1':
            psi0 = tensor(basis(2, 1), basis(2, 0))
        elif self.excited_qubit == 'q2':
            psi0 = tensor(basis(2, 0), basis(2, 1))

        options = Options(nsteps=10000)
        
        # to take into account the interaction before initial time
        t0, t1 = time[0], time[-1]
        dt = time[1] - time[0]
        initial_time = np.arange(0.0, t0, dt)
        
        extended_time = np.append(initial_time, time)
        # time evolution with various detunings
        if self.tuned_qubit == 'q1':
            for idx, Delta in enumerate(Delta_list):
                H = Delta * sp1 * sm1 + Hint

                result = mesolve(H, psi0, extended_time, c_ops=c_ops, e_ops=e_ops,
                                 options=options)
        
                result_arr[idx, :] = result.expect[0][len(initial_time):]
        elif self.tuned_qubit == 'q2':
            for idx, Delta in enumerate(Delta_list):
                H = Delta * sp2 * sm2 + Hint

                result = mesolve(H, psi0, extended_time, c_ops=c_ops, e_ops=e_ops,
                                 options=options)
        
                result_arr[idx, :] = result.expect[0][len(initial_time):]

        return r0 + r1 * result_arr

    def _guess_init_params(self):
        
        sig_var = np.var(self.signal, axis=1)
        i0 = np.argmax(sig_var)
        amp0, sig0 = self.amp[i0], self.signal[i0]
        sig0_rfft = np.fft.rfft(sig0 - np.mean(sig0))
        sig0_rfftfreq = np.fft.rfftfreq(len(sig0), d=self.time[1]-self.time[0])
        
        # initial guess of the oscillation frequency
        f0 = sig0_rfftfreq[np.argmax(np.abs(sig0_rfft))]
        # fit sig0 with damped oscillation curve
        def damped_osc(time, a, b, t0, gamma, f):
            return a * np.cos(2 * np.pi * f * (time - t0)) * np.exp(-gamma * time) + b
        
        if (np.max(sig0) - sig0[0]) < (sig0[0] - np.min(sig0)):
            a0 = 0.5 * (np.max(sig0) - np.min(sig0))
        else:
            a0 = - 0.5 * (np.max(sig0) - np.min(sig0))

        popt, pcov = curve_fit(damped_osc, self.time, sig0,
                               p0=[a0, np.mean(sig0), 0.0, 0.0, f0])

        # check the fit quality
        if np.abs(popt[0]) < 0.1 * (np.max(sig0) - np.min(sig0)):
            gamma0 = 1 / (self.time[-1] - self.time[0])
        else:
            f0 = popt[-1]
            if popt[-2] > 0.0:
                gamma0 = popt[-2]
            else:
                gamma0 = 1 / (self.time[-1] - self.time[0])
        
        # convert oscillation freq to g
        g0 = 2 * np.pi * f0 / 2

        min_, max_ = np.min(self.signal), np.max(self.signal)
        if self.excited_qubit == self.measured_qubit:
            # starting from excited state
            if (sig0[0] - min_) > (max_ - sig0[0]):
                r1 = max_ - min_
                r0 = min_
            else:
                r1 = min_ - max_
                r0 = max_
        else:
            # starting from ground state
            if (sig0[0] - min_) > (max_ - sig0[0]):
                r1 = min_ - max_
                r0 = max_
            else:
                r1 = max_ - min_
                r0 = min_

        sig_var_mid = 0.5 * (np.max(sig_var) + np.min(sig_var))
        i1 = np.argmin(np.abs(sig_var[i0:] - sig_var_mid)) + i0
        i2 = np.argmin(np.abs(sig_var[:i0] - sig_var_mid))
        c1 = 2 * g0 / (self.amp[i1] - self.amp[i2])

        self.p0 = [g0, gamma0, gamma0, gamma0, gamma0] + [r0, r1, amp0, c1]
        self.p0 = self.p0 + [0.0] * (self.amp_polyorder - 1)

    def analyze(self, p0=None, plot=True):

        self._set_init_params(p0)

        def lsq_func(params):        
            result_arr = self.fit_func(self.time, self.amp, params)
            
            return (result_arr - self.signal).flatten()

        res = least_squares(
            lsq_func, self.p0,
            bounds=([0.0, 0.0, 0.0, 0.0, 0.0, -np.inf, -np.inf, -np.inf] +
                    [-np.inf] * self.amp_polyorder, np.inf),
            ftol=1e-12, xtol=1e-12)

        self.popt = res.x

        self._get_pcov(res)
        self.perr = np.sqrt(np.diag(self.pcov))

        self.g = self.popt[0]
        self.g_sigma_err = self.perr[0]

        self.Gamma1_q1 = self.popt[1]
        self.Gamma1_q1_sigma_err = self.perr[1]
        self.Gamma1_q2 = self.popt[2]
        self.Gamma1_q2_sigma_err = self.perr[2]

        self.Gamma_phi_q1 = self.popt[3]
        self.Gamma_phi_q1_sigma_err = self.perr[3]
        self.Gamma_phi_q2 = self.popt[4]
        self.Gamma_phi_q2_sigma_err = self.perr[4]

        self.amp0 = self.popt[7]
        self.amp0_sigma_err = self.perr[7]
        
        dz = self.amp - self.amp0
        c = self.popt[8:]
        self.detuning = np.sum([c[i] * dz ** (i + 1) for i in range(len(c))],
                               axis=0) / (2 * np.pi)

        if plot:
            self.plot_result()

    def _get_pcov(self, res): 
        # Do Moore-Penrose inverse discarding zero singular values.
        _, s, VT = svd(res.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        self.pcov = np.dot(VT.T / s**2, VT)
    
    def plot_result(self):
        # rescale data axes
        _, self.time_prefix = number_with_si_prefix(np.max(np.abs(self.time)))
        self.time_scaler = si_prefix_to_scaler(self.time_prefix)

        _, self.amp_prefix = number_with_si_prefix(np.max(np.abs(self.amp)))
        self.amp_scaler = si_prefix_to_scaler(self.amp_prefix)

        fig = plt.figure()
        # plot data
        plt.subplot(3, 1, 1)
        plt.pcolor(self.time / self.time_scaler,
                   self.amp / self.amp_scaler, self.signal, shading='auto',
                   cmap=plt.cm.RdBu_r)
        plt.axhline(y=self.amp0 / self.amp_scaler, ls='--', color='black')
        plt.ylabel('Amp' +
                   (' (' + self.amp_prefix + ')' if len(self.amp_prefix) > 0 else ''),
                   fontsize='small')
        plt.xlim(np.min(self.time) / self.time_scaler, np.max(self.time) / self.time_scaler)
        plt.tick_params(axis='x', which='both', labelbottom=False)
        plt.yticks(fontsize='x-small')
        plt.twinx()
        plt.ylabel('Data', fontsize='medium')
        plt.tick_params(axis='y', which='both', right=False, labelright=False)

        # plot fit      
        plt.subplot(3, 1, 2)

        fit_time = np.linspace(np.min(self.time), np.max(self.time), 100)
        fit_amp = np.linspace(np.min(self.amp), np.max(self.amp), 100)
        
        plt.pcolor(fit_time / self.time_scaler,
                   fit_amp / self.amp_scaler,
                   self.fit_func(fit_time, fit_amp, self.popt),
                   shading='auto', cmap=plt.cm.RdBu_r)

        plt.axhline(y=self.amp0 / self.amp_scaler, ls='--', color='black')
        plt.xlim(np.min(self.time) / self.time_scaler, np.max(self.time) / self.time_scaler)
        
        _, g_2pi_prefix = number_with_si_prefix(np.max(np.abs(self.g / (2 * np.pi))))
        g_2pi_scaler = si_prefix_to_scaler(g_2pi_prefix)

        plt.tick_params(axis='x', which='both', labelbottom=False)
        plt.yticks(fontsize='x-small')
        plt.ylabel('Amp', fontsize='small')

        plt.twinx()
        plt.ylabel('Fit', fontsize='medium')
        plt.tick_params(axis='y', which='both', right=False, labelright=False)

        plt.subplot(3, 1, 3)
        
        fit_data = self.fit_func(fit_time, self.amp, self.popt)

        for i in range(len(self.amp)):
            plt.plot(self.time / self.time_scaler, self.signal[i, :], '.', color=f'C{i}')
            plt.plot(fit_time / self.time_scaler, fit_data[i, :], '-', color=f'C{i}')
        plt.xlabel('Interaction Time (' + self.time_prefix + 's)', fontsize='small')
        plt.ylabel('Signal', fontsize='small')
        plt.xticks(fontsize='x-small')
        plt.yticks(fontsize='x-small')
        plt.xlim(np.min(self.time) / self.time_scaler, np.max(self.time) / self.time_scaler)

        _fit_result_msg = [
            (r'$g/2\pi=%.3f\pm %.3f$' % (self.g / (2 * np.pi * g_2pi_scaler),
                                         self.g_sigma_err / (2 * np.pi * g_2pi_scaler))) +
            ' ' + g_2pi_prefix + 'Hz',
            ('Amp0$=%.3f\pm %.3f$' % (self.amp0 / self.amp_scaler,
                                      self.amp0_sigma_err / self.amp_scaler)) +
            ((' ' + self.amp_prefix) if len(self.amp_prefix) > 0 else '')
            ]
        plt.suptitle('Vacuum Rabi Osc.: ' +
                     ', '.join(_fit_result_msg), fontsize='medium')
        
        plt.tight_layout()
    # def fit_func(self):
        