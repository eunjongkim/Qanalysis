import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import windows

def _get_envelope(s, t, f):
    """
    Find the envelope of an oscillating real-valued signal `s` as a function of time `t` by demodulating at the
    frequency specified by `f` followed by low-pass filtering at cutoff of half the specified frequency.
    """
    # time step
    dt = t[1] - t[0]

    # perform manual demodulation to get envelope
    I = s * np.cos(2 * np.pi * f * t)
    Q = s * np.sin(2 * np.pi * f * t)

    # extract envelope by low-pass filtering at cutoff of f / 2
    _window = int(1 / (f * dt))
    _hann = windows.hann(_window * 2, sym=True)
    _hann = _hann / np.sum(_hann)

    envI = np.convolve(I, _hann, 'same') * 2
    envQ = np.convolve(Q, _hann, 'same') * 2

    return envI, envQ

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
        popt, pcov = curve_fit(self.fit_func, self.time, self.signal, p0=self.p0, **kwargs)
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
        return a / 2 * (1 - np.cos(np.pi * amp / (amp_pi))) + b

    def _guess_init_params(self):
        # perform fft to find frequency of Rabi oscillation
        freq = np.fft.rfftfreq(len(self.signal), d=(self.amp[1] - self.amp[0]))
        # initial parameter for Rabi frequency from FFT
        F0 = freq[np.argmax(np.abs(np.fft.rfft(self.signal - np.mean(self.signal))))]
        amp_pi0 = 1 / (2 * F0)
        b0 = self.signal[0]
        a0 = np.max(self.signal) - np.min(self.signal)
        if np.abs(b0 - np.min(self.signal)) > np.abs(b0 - np.max(self.signal)):
            a0 *= -1

        self.p0 = [amp_pi0, a0, b0]

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
        popt, pcov = curve_fit(self.fit_func, self.amp, self.signal, p0=self.p0, **kwargs)
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
        plt.legend(loc=0, fontsize=14)

        amp_fit = np.linspace(self.amp[0], self.amp[-1], fit_n_pts)

        plt.plot(amp_fit, self.fit_func(amp_fit, *(self.popt)), label="Fit", lw=2, color="red")
        plt.title(r"$a_{\pi} = %.5f \pm %.5f$" % (self.amp_pi,
                                                    2 * self.amp_pi_sigma_err))
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
        T10 = (self.time[-1] - self.time[0]) / 3  # choose 1/3 of time window as init. value
        self.p0 = [T10, a0, b0]

    def _save_fit_results(self, popt, pcov):
        super()._save_fit_results(popt, pcov)

        self.T1 = popt[0]
        self.T1_sigma_err = np.sqrt(pcov[0, 0])
    def plot_result(self, fit_n_pts=1000):
        super().plot_result()

        # get most of the plotting done
        fig = self._plot_base()

        time_fit = np.linspace(self.time[0], self.time[-1], fit_n_pts)
        plt.plot(time_fit / 1e-6, self.fit_func(time_fit, *(self.popt)), label="Fit", lw=2, color="red")
        plt.title(r"$T_1 = %.5f \pm %.5f \:\mu\mathrm{s}$" % (self.T1 / 1e-6,
                                                            2 * self.T1_sigma_err / 1e-6))
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

    def fit_func(self, t, T2, δf, t0, a, b):
        """
        Fitting Function for Ramsey Fringes
            f(t) = a exp(-t/T2) cos[2π∆f(t-t0)] + b
        """
        return a * np.exp(- t / T2) * np.cos(2 * np.pi * δf * (t - t0)) + b

    def _guess_init_params(self):
        b0 = np.mean(self.signal)
        signal0 = self.signal - b0
        a0 = np.max(np.abs(signal0))

        # perform fft to find frequency of Ramsey fringes
        freq = np.fft.rfftfreq(len(self.signal),
                               d=(self.time[1]-self.time[0]))
        δf0 = freq[np.argmax(np.abs(np.fft.rfft(self.signal - np.mean(self.signal))))]

        # in-phase and quadrature envelope
        envI, envQ = _get_envelope(signal0, self.time, δf0)
        t00 = np.arctan(np.sum(envQ) / np.sum(envI)) / (2 * np.pi * δf0)

        # extract envelope by low-pass filtering at cutoff of δf0 / 2
        _window = int(1 / (δf0 * (self.time[1] - self.time[0])))
        _hann = windows.hann(_window * 2, sym=True)
        _hann = _hann / np.sum(_hann)

        env = np.sqrt(envI ** 2 + envQ ** 2)
        if env[-1] < env[0]: # sanity check: make sure the envelope is decreasing over time
            T20 = - (self.time[-1] - self.time[0]) / np.log(env[-1] / env[0])
        else:
            T20 = self.time[-1] - self.time[0]

        self.p0 = [T20, δf0, t00, a0, b0]

    def _save_fit_results(self, popt, pcov):
        super()._save_fit_results(popt, pcov)

        self.T2Ramsey = popt[0]
        self.T2Ramsey_sigma_err = np.sqrt(pcov[0, 0])
        self.delta_freq = popt[1]
        self.delta_freq_sigma_err = np.sqrt(pcov[1, 1])

    def plot_result(self, fit_n_pts=1000):
        super().plot_result()

        # get most of the plotting done
        fig = self._plot_base()

        time_fit = np.linspace(self.time[0], self.time[-1], fit_n_pts)

        plt.plot(time_fit / 1e-6, self.fit_func(time_fit, *(self.popt)),
                 label="Fit", lw=2, color="red")
        plt.title(r"$T_2^* = %.5f \pm %.5f\:\mu\mathrm{s}$, $\Delta f = %.4f \pm %.4f \:\mathrm{MHz}$"\
                  % (self.T2Ramsey / 1e-6, 2 * self.T2Ramsey_sigma_err / 1e-6,
                     self.delta_freq / 1e6, 2 * self.delta_freq_sigma_err / 1e6))
        fig.tight_layout()
        plt.show()

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

class SpinEcho(TimeDomain):
    """
    Class to analyze and visualize spin echo decay (T2 Hahn echo experiment) data.
    """
    def __init__(self, time, signal):
        super().__init__(time, signal)
        self.T2Echo = None

    def fit_func(self, t, T2, a, b):
        """
        Fitting Function for T2
        """
        return a * np.exp(- t / T2) + b

    def _guess_init_params(self):

        a0 = self.signal[0] - self.signal[-1]
        b0 = self.signal[-1]
        T20 = (self.time[-1] - self.time[0]) / 3  # choose 1/3 of time window as init. value
        self.p0 = [T20, a0, b0]

    def _save_fit_results(self, popt, pcov):
        super()._save_fit_results(popt, pcov)
        self.T2Echo = popt[0]
        self.T2Echo_sigma_err = np.sqrt(pcov[0, 0])

    def plot_result(self, fit_n_pts=1000):
        super().plot_result()

        # get most of the plotting done
        fig = self._plot_base()

        time_fit = np.linspace(self.time[0], self.time[-1], fit_n_pts)
        plt.plot(time_fit / 1e-6, self.fit_func(time_fit, *(self.popt)), label="Fit", lw=2, color="red")
        plt.title(r"$T_{2E} = %.5f \pm %.5f \:\mu\mathrm{s}$" % (self.T2Echo / 1e-6,
                                                                  2 * self.T2Echo_sigma_err / 1e-6))
        fig.tight_layout()
        plt.show()

class EasyReadout:
    """
    Class to implement easy analysis of readout data.
    """
    def __init__(self, data, readout_type='phase', ang_tol=np.pi/1000):
        self.data = data
        self.n_pts = len(self.data)

        # This will be overwritten in `self._project_to_line` function call
        self.n = 1.0 + 0.0j     # unit vector connecting the ground and excited state points in the complex plane
        self.v_g = 0.0 + 0.0j   # ground state point in the complex plane
        self.v_e = 0.0 + 0.0j   # excited state point in the complex plane

        # qubit population extracted by fitting to a line and
        self.population = None
        self._project_to_line(readout_type, ang_tol)

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
        fit_axes[1].legend(loc=1, fontsize='x-small')

