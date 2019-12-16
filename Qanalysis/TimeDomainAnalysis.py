# classes for implementing analysis on time-domain measurement data
# written by EK, XZ

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
import matplotlib.pyplot as plt

class TimeDomainAnalysis:
    def __init__(self, t_data, v_data, readout_type='phase', N=1000):
        # initialize parameters
        self.time = t_data
        self.data = v_data
        self.n_pts = len(self.data)
        self.is_analyzed = False

        # will be overwritten in `_project_to_line` function call
        self.n = 1.0 + 0.0j
        self.v_g = 0.0 + 0.0j
        self.v_e = 0.0 + 0.0j

        # qubit population extracted by fitting to a line and
        self.population = None
        self._project_to_line(readout_type, N)

    def _project_to_line(self, readout_type, N):
        if readout_type not in ['magnitude', 'phase']:
            raise ValueError(\
                "The `readout_type` must be either `phase` or `magnitude`")
        self.readout_type = readout_type

        # fit a straight line on the full data

        A, B, C = self._fit_line(N)

        # find (approximate) ground state and excited state voltage
        if readout_type == 'magnitude':
            mag_data = np.abs(self.data)
            self.v_g = self._find_point_closest(\
                    self.data[np.argmin(mag_data)], A, B, C)
            self.v_e = self._find_point_closest(\
                    self.data[np.argmax(mag_data)], A, B, C)
        elif readout_type == 'phase':
            phase_data = np.unwrap(np.angle(self.data))
            self.v_g = self._find_point_closest(\
                    self.data[np.argmin(phase_data)], A, B, C)
            self.v_e = self._find_point_closest(\
                    self.data[np.argmax(phase_data)], A, B, C)

        # unit vector along the fitted line
        self.n = (self.v_e - self.v_g) / np.abs(self.v_e - self.v_g)

        # projection of complex data to a line v_orig + x * n
        self.population = (self._inner_product(self.data - self.v_g, self.n) /
                           np.abs(self.v_e - self.v_g))

    def _inner_product(self, z1, z2):
        """
        Element-wise inner product between complex vectors or between a complex
        vector and a complex number.
        """
        return z1.real * z2.real + z1.imag * z2.imag


    def _fit_line(self, N):
        """
        Fit a straight line to a complex dataset such that standard deviation
        of data projected onto the line is minimized. `N` determines the
        precision in the angle of the new axis.
        """
        v = self.data

        θ = np.linspace(0, np.pi, N)

        σ, μ = np.zeros(N), np.zeros(N)
        for m, θ_ in enumerate(θ):
            # projection of data points to axis rotated by θ from the real axis
            x = self._inner_product(v, np.cos(θ_) + 1j * np.sin(θ_))
            σ[m] = np.std(x)
            μ[m] = np.mean(x)

        # find the angle with minimum standard deviation of data
        m_min = np.argmin(σ)
        θ_min, μ_min = θ[m_min], μ[m_min]
        
        v0 = μ_min * (np.cos(θ_min) + 1j * np.sin(θ_min))

        A, B, C = np.cos(θ_min), np.sin(θ_min), - μ_min

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

    def analyze(self):
        """
        Will be overwritten in subclass
        """

    def _plot_base(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].scatter(self.data.real, self.data.imag, s=2,
                                label="Data", color="blue")
        rmin, rmax = np.min(self.data.real), np.max(self.data.real)
        imin, imax = np.min(self.data.imag), np.max(self.data.imag)

        center = (rmin + rmax) / 2 + (imin + imax) / 2 * 1j
        window = np.max([rmax - rmin, imax - imin]) * 1.1

        axes[0].set_xlim([center.real - window / 2, center.real + window / 2])
        axes[0].set_ylim([center.imag - window / 2, center.imag + window / 2])

        t_max = np.sqrt((rmax - rmin) ** 2 + (imax - imin) ** 2)
        v_ax = (0.5 * (self.v_e + self.v_g) +
                np.linspace(-t_max / 2, t_max / 2, 100) * self.n)

        # plot projection axis
        axes[0].plot(v_ax.real, v_ax.imag, ls='--', color='black')
        v_dec = (0.5 * (self.v_g + self.v_e) +
                 np.linspace(-t_max / 2, t_max / 2, 100) *
                 (self.n.imag - 1j * self.n.real))

        # plot decision boundary
        axes[0].plot(v_dec.real, v_dec.imag, ls='-', color='black', lw=2)

        axes[0].set_xlabel("I", fontsize=16)
        axes[0].set_ylabel("Q", fontsize=16)
        axes[0].legend(loc=0, fontsize=16)

        # plot data
        axes[1].scatter(self.time/1e-6, self.population, label="Data",
                        color="black", s=2)
        axes[1].set_xlabel(r"Time ($\mu$s)", fontsize=16)
        axes[1].set_ylabel("Population", fontsize=16)
        axes[1].legend(loc=0, fontsize=14)
        axes[1].set_ylim([-0.1, 1.1])

        fig.tight_layout()
        return fig, axes

    def plot_result(self):
        """
        Will be overwritten in subclass
        """
        if not self.is_analyzed:
            raise ValueError("The data must be analyzed before plotting")

class T1Analysis(TimeDomainAnalysis):
    """
    class to perform analysis and visualize population decay data
    """
    def __init__(self, t_data, v_data, readout_type='phase', N=1000):
        super().__init__(t_data, v_data, readout_type=readout_type, N=1000)
        self.T1 = None
        self.T1_popt = None

    def fit_func(self, t, T1, a, b):
        """
        Fitting Function for T1
        """
        return a * np.exp(- t / T1) + b

    def analyze(self, p0=[1.0e-6, 1.0, 0.0]):
        """
        Plot population decay curve with model
            p(t) = A * e^(-t / T1) + B
        `p0`: initial parameter [T1, A, B] for the fit
        """

        popt, pcov = curve_fit(self.fit_func, self.time, self.population,
                               p0=p0, maxfev=100000)
        self.T1 = popt[0]
        self.T1_popt = popt

        self.is_analyzed = True
        return popt, pcov

    def plot_result(self):
        super().plot_result()

        # get most of the plotting done
        fig, axes = self._plot_base()

        time_fit = np.linspace(self.time[0], self.time[-1], 1000)

        axes[1].plot(time_fit/1e-6,
                     self.fit_func(time_fit, *(self.T1_popt)),
                     label="Fit", lw=2, color="red")
        axes[1].set_title(r"$T_1 = %.5f \mu\mathrm{s}$" % (self.T1/1e-6),
                          fontsize=16)
        fig.tight_layout()

class T2RamseyAnalysis(TimeDomainAnalysis):
    """
    class to perform analysis and visualize Ramsey fringes data
    """

    def __init__(self, t_data, v_data, readout_type='phase', N=1000):
        super().__init__(t_data, v_data, readout_type=readout_type, N=1000)
        self.T2Ramsey = None
        self.T2Ramsey_popt = None
        self.T2Ramsey_pcov = None

    def fit_func(self, t, T2, δf, t0, a, b, c, Td):
        """
        Fitting Function for Ramsey Fringes
        """
        return (a * np.exp(- t / T2) * np.cos(2 * np.pi * δf * (t - t0)) +
                b + c * np.exp(-t / Td))

    def analyze(self, p0=None):
        """
        Plot Ramsey fringes curve with model
            p(t) = a * e^(-t / T2) * cos(2π δf (t - t0)) + b + c e^(-t/Td)
        `p0`: optional initial parameter [T2, δf, t0, a, b, c, Td] for the fit
        """

        if p0 is None:
            p0 = [1.0e-6, 1.0e6, 0.0, 0.5, 0.5, 0.0, 1.0e-6]
            # perform fft to find frequency of Ramsey fringes
            freq = np.fft.rfftfreq(len(self.population), d=self.time[1]-self.time[0])
            δf0 = freq[np.argmax(np.abs(np.fft.rfft(self.population - np.mean(self.population))))]

            p0[1] = δf0

        popt, pcov = curve_fit(self.fit_func, self.time,
                               self.population, p0=p0, maxfev=100000)
        self.T2Ramsey = popt[0]
        self.T2Ramsey_popt = popt
        self.T2Ramsey_pcov = pcov

        self.is_analyzed = True
        return popt, pcov

    def plot_result(self):
        super().plot_result()

        # get most of the plotting done
        fig, axes = self._plot_base()

        time_fit = np.linspace(self.time[0], self.time[-1], 1000)

        axes[1].plot(time_fit/1e-6,
                     self.fit_func(time_fit, *self.T2Ramsey_popt),
                     label="Fit", lw=2, color="red")
        axes[1].set_title(r"$T_2^* = %.5f \mu\mathrm{s}$" % (self.T2Ramsey/1e-6),
                          fontsize=16)
        fig.tight_layout()

class RabiAnalysis(TimeDomainAnalysis):
    """
    class to perform analysis and visualize Rabi oscillation data
    """
    def __init__(self, t_data, v_data, readout_type='phase', N=1000):
        super().__init__(t_data, v_data, readout_type=readout_type, N=1000)
        self.TRabi = None
        self.TRabi_popt = None

    def fit_func(self, t, T2, T1, t0, F0, δf, A, B, C):
        """
        The strongly-driven Rabi oscillation curve based on Torrey's solution
        A e^(-at) + B e^(-b t) cos(Ω(t-t0)) + C

        a = 1/T2 - (1/T2 - 1/T1) Δ²/(Δ² + Ω₀²) + O(α³)
        b = 1/T2 - 1/2(1/T2 - 1/T1) Ω₀²/(Δ² + Ω₀²) + O(α³)
        Ω = √(Δ² + Ω₀²) + O(α²)
        where α ≡ (1 / T₂ - 1 / T₁) / Ω₀ ≪ 1 is a small parameter when damping
        rates (1/T₂, 1/T₁) are very small compared to the Rabi frequency Ω₀=2πF₀.
        """
        # convert frequencies to angular frequencies
        Ω0 = 2 * np.pi * F0
        Δ = 2 * np.pi * δf

        # constants
        a = 1 / T2 - (1 / T2 - 1 / T1) * Δ ** 2 / (Δ ** 2 + Ω0 ** 2)
        b = 1 / T2 - 0.5 * (1 / T2 - 1 / T1) * Δ ** 2 / (Δ ** 2 + Ω0 ** 2)
        Ω = np.sqrt(Δ ** 2 + Ω0 ** 2)
        return (A * np.exp(- a * t) +
                B * np.exp(- b * t) * np.cos(Ω * (t - t0)) + C)

    def analyze(self, p0=None):
        """
        Plot Rabi oscillation curve with model
        """
        # perform fft to find frequency of Rabi oscillation
        freq = np.fft.rfftfreq(len(self.population),
                               d=self.time[1]-self.time[0])
        
        if p0 is None:
            p0 = []
            # initial parameter for Rabi frequency from FFT
            F0 = freq[np.argmax(np.abs(np.fft.rfft(self.population - np.mean(self.population))))]
        


    def plot_result(self):
        super().plot_result()

        # get most of the plotting done
        fig, axes = self._plot_base()

        time_fit = np.linspace(self.time[0], self.time[-1], 1000)

        axes[1].plot(time_fit/1e-6,
                     self.fit_func(time_fit, *self.T2Ramsey_popt),
                     label="Fit")
        axes[1].set_title(r"$T_2^* = %.5f \mu\mathrm{s}$" % (self.T2Ramsey/1e-6),
                          fontsize=16)
        fig.tight_layout()


class ReadoutAnalysis:
    def __init__(self, g_data, e_data):
        # initialize parameters
        self.g_data = g_data
        self.e_data = e_data
        self.is_analyzed = False

        self.n_shots = len(self.g_data)

        self.v_orig = None
        # direction from g to e, normal to decision boundary
        self.n = None

        self.x_g = None
        self.x_e = None

        self.x0 = None
        # a point in the decision boundary
        self.v_dec = None

    def analyze(self, N=10000):
        """
        Analyze the single-shot readout result
        """
        # fit a straight line on the full data
        A, B, C = self._fit_line(N)

        v_g = self._find_point_closest(np.mean(self.g_data), A, B, C)
        v_e = self._find_point_closest(np.mean(self.e_data), A, B, C)

        # origin of a new coordinate along the fitted line
        self.v_orig = 0.5 * (v_g + v_e)
        # unit vector along the fitted line
        self.n = (v_e - v_g) / np.abs(v_e - v_g)

        # projection of complex data to a line v_orig + x * n
        self.x_g = self._inner_product(self.g_data - self.v_orig, self.n)
        self.x_e = self._inner_product(self.e_data - self.v_orig, self.n)

    def _plot_base(self, bins=50):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        # plot distribution of data with decision boundary
        axes[0].scatter(self.g_data.real, self.g_data.imag, s=2,
                        label=r"$|g\rangle$", color="blue", alpha=0.5)
        axes[0].scatter(self.e_data.real, self.e_data.imag, s=2,
                        label=r"$|e\rangle$", color="red", alpha=0.5)

        data = np.append(self.g_data, self.e_data)
        rmin, rmax = np.min(data.real), np.max(data.real)
        imin, imax = np.min(data.imag), np.max(data.imag)

        center = (rmin + rmax) / 2 + (imin + imax) / 2 * 1j
        window = np.max([rmax - rmin, imax - imin]) * 1.1

        axes[0].set_xlim([center.real - window / 2, center.real + window / 2])
        axes[0].set_ylim([center.imag - window / 2, center.imag + window / 2])

        t_max = np.sqrt((rmax - rmin) ** 2 + (imax - imin) ** 2)
        v_ax = self.v_orig + np.linspace(-t_max / 2, t_max / 2, 100) * self.n

        # plot projection axis
        axes[0].plot(v_ax.real, v_ax.imag, ls='--', color='black')
        v_dec = (self.v_dec +
                 np.linspace(-t_max / 2, t_max / 2, 100) *
                 (self.n.imag - 1j * self.n.real))
        # plot decision boundary
        axes[0].plot(v_dec.real, v_dec.imag, ls='-', color='black', lw=2)

        axes[0].set_xlabel("I", fontsize=16)
        axes[0].set_ylabel("Q", fontsize=16)
        axes[0].legend(loc=0, fontsize=16)

        # plot the histogram
        axes[1].hist(self.x_g, bins=bins, label=r"$|g\rangle$", color="blue", alpha=0.5)
        axes[1].hist(self.x_e, bins=bins, label=r"$|e\rangle$", color="red", alpha=0.5)
        axes[1].axvline(self.x0, color='black', lw=2)
        axes[1].set_xlabel("$x$", fontsize=16)
        axes[1].set_ylabel("Number of occurrences", fontsize=16)
        axes[1].legend(loc=0, fontsize=16)
        fig.tight_layout()
        return fig, axes

    def plot_result(self, bins=50):
        if not self.is_analyzed:
            raise ValueError("The data first needs to be analyzed. Perforom `.analyze()` first.")

    ## TODO: fit bimodal Gaussian


    def _fit_line(self, N):
        """
        Fit a straight line to a complex dataset such that standard deviation
        of data projected onto the line is minimized. `N` determines the
        precision in the angle of the new axis.
        """
        v = np.append(self.g_data, self.e_data)

        θ = np.linspace(0, np.pi / 2, N)
        σ, μ = np.zeros(N), np.zeros(N)
        for m, θ_ in enumerate(θ):
            # projection of data points to axis rotated by θ from the real axis
            x = self._inner_product(v, np.cos(θ_) + 1j * np.sin(θ_))
            σ[m] = np.std(x)
            μ[m] = np.mean(x)

        # find the angle with minimum standard deviation of data
        m_min = np.argmin(σ)
        θ_min, μ_min = θ[m_min], μ[m_min]

        v0 = μ_min * (np.cos(θ_min) + 1j * np.sin(θ_min))

        A, B, C = np.cos(θ_min), np.sin(θ_min), - μ_min

        return A, B, C


    def _inner_product(self, z1, z2):
        """
        Element-wise inner product between complex vectors or between a complex
        vector and a complex number.
        """
        return z1.real * z2.real + z1.imag * z2.imag


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

class SingleShotReadoutAnalysis(ReadoutAnalysis):
    """
    Analyze the single-shot measurement result.
    """
    def __init__(self, g_data, e_data):
        super().__init__(g_data, e_data)
        self.fidelity = None

    def analyze(self, N=10000):
        super().analyze(N=N)
        # find decision boundary that minimizes the infidelity
        find_boundary = minimize_scalar(self._infidelity, options={'xtol': 1.48e-08, 'maxiter': 10000})

        # location of the decision boundary in the x-coordinate
        self.x0 = find_boundary.x
        # location of decision boundary in the complex plane
        self.v_dec = self.v_orig + self.x0 * self.n
        # Readout fidelity for the optimal decision boundary
        self.fidelity = 1 - find_boundary.fun
        self.is_analyzed = True
    # def _fit_double_gaussian(self, bins=50):
    #
    #     gauss = lambda x, μ, σ, A: A * np.exp(- (x - μ) ** 2 / (2 * σ ** 2))
    #     bimodal = lambda x, μ1, σ1, A1, μ2, σ2, A2: (gauss(x, μ1, σ1, A1) +
    #                                                  gauss(x, μ2, σ2, A2))
    #
    #     x_data = np.append(self.x_g, self.x_e)
    #     y, bin_edges = np.hist(x_data, bins=bins)
    #     x = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
    #
    #     y_g =
    #
    #     μ10 =
    #     popt, cov = curve_fit(bimodal, x, y, p0=[])
    def plot_result(self, bins=50):
        super().plot_result(bins=bins)

        fig, axes = self._plot_base(bins=bins)
        axes[1].set_title("x0 = %.4f, F = %.4f" % (self.x0, self.fidelity), fontsize=16)

    def _infidelity(self, x0):
        """
        Readout infidelity when a decision boundary is drawn at x = x0.
        1 - F = P(g|e) + P(e|g)
        """
        return np.mean(self.x_g > x0) + np.mean(self.x_e < x0)

class MultiShotReadoutAnalysis(ReadoutAnalysis):
    def __init__(self, g_data, e_data):
        super().__init__(g_data, e_data)
        self.v_g = np.mean(g_data)
        self.v_e = np.mean(e_data)

    def analyze(self, N=10000):
        """
        Location of the decision boundary in the x-coordinate. for multi-shot,
        choose the midpoint between average of ground state data and excited
        state data.
        """
        v_g, v_e = self.v_g, self.v_e

        # origin of a new coordinate along the fitted line
        self.v_orig = 0.5 * (v_g + v_e)
        # unit vector along the fitted line
        self.n = (v_e - v_g) / np.abs(v_e - v_g)

        # projection of complex data to a line v_orig + x * n
        self.x_g = self._inner_product(self.g_data - self.v_orig, self.n)
        self.x_e = self._inner_product(self.e_data - self.v_orig, self.n)

        self.x0 = 0.5 * (np.mean(self.x_g) + np.mean(self.x_e))

        # location of decision boundary in the complex plane
        self.v_dec = self.v_orig + self.x0 * self.n
        self.is_analyzed = True

    def plot_result(self, bins=50):
        super().plot_result(bins=bins)

        fig, axes = self._plot_base(bins=bins)
        axes[0].scatter(self.v_g.real, self.v_g.imag, s=40)
        axes[0].scatter(self.v_e.real, self.v_e.imag, s=40)
