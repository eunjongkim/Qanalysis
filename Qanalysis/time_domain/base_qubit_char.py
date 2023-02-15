from Qanalysis.helper_functions import UnitfulNumber, get_envelope
from Qanalysis.time_domain.base_time_domain import AmpSweep, TimeSweep
import matplotlib.pyplot as plt
import numpy as np

class Rabi(TimeSweep):
    """
    Class to analyze and visualize Rabi oscillation data
    """
    def fit_func(
        self, t: np.ndarray,
        TRabi: float, F: float, t0: float, Td: float, a: float, b: float, c: float
    ) -> np.ndarray:
        """
        The strongly-driven Rabi oscillation curve based on Torrey's solution
        a e^(-t/Td) + b e^(-t/TRabi) cos(2π F(t-t0)) + c

        1/Td = 1/T2 - (1/T2 - 1/T1) Δ²/(Δ² + Ω₀²) + O(α³)
        1/TRabi = 1/T2 - 1/2(1/T2 - 1/T1) Ω₀²/(Δ² + Ω₀²) + O(α³)
        Ω = √(Δ² + Ω₀²) + O(α²)
        where α ≡ (1 / T₂ - 1 / T₁) / Ω₀ ≪ 1 is a small parameter when damping
        rates (1/T₂, 1/T₁) are very small compared to the Rabi frequency Ω₀=2πF₀.
        """
        return (
            a * np.exp(- t / Td) +
            b * np.exp(- t / TRabi) * np.cos(2 * np.pi * F * (t - t0)) +
            c
        )

    def _guess_init_params(self) -> None:
        # perform fft to find frequency of Rabi oscillation
        freq: np.ndarray = np.fft.rfftfreq(
            len(self.signal), d=(self.time[1] - self.time[0])
        )
        # initial parameter for Rabi frequency from FFT
        F0: float = freq[
            np.argmax(np.abs(np.fft.rfft(self.signal - np.mean(self.signal))))
        ]

        a0 = np.max(np.abs(self.signal - np.mean(self.signal)))
        b0 = 0.0
        c0 = np.mean(self.signal)
        t0 = 0.0
        T0 = self.time[-1] - self.time[0]

        self.p0 = [T0, F0, t0, T0, a0, b0, c0]

    def _save_fit_results(self, popt: np.ndarray, pcov: np.ndarray) -> None:
        super()._save_fit_results(popt, pcov)

        self.TRabi: float = popt[0]
        self.RabiFrequency: float = popt[1]
        self.TRabi_sigma_err: float = np.sqrt(pcov[0, 0])
        self.RabiFrequency_sigma_err: float = np.sqrt(pcov[1, 1])

    def plot_result(self, fit_n_pts: int = 1000) -> plt.Figure:
        fig = super().plot_result(fit_n_pts = fit_n_pts)

        TRabi_unitful = UnitfulNumber(
            self.TRabi, error = self.TRabi_sigma_err, base_unit = self.time_unit
        )
        OmegaRabi_2pi_unitful = UnitfulNumber(
            self.OmegaRabi / (2 * np.pi),
            error = self.RabiFrequency_sigma_err / (2 * np.pi),
            base_unit = 'Hz'
        )
        plt.title(
            f"$T_R$ = {TRabi_unitful}, $\Omega_R/2\pi$ = {OmegaRabi_2pi_unitful}"
        )
        fig.tight_layout()
        plt.show()
        return fig


class PopulationDecay(TimeSweep):
    """
    Class to analyze and visualize population decay (T1 experiment) data.
    """
    def fit_func(
        self, t: np.ndarray, T1: float, a: float, b: float
    ) -> np.ndarray:
        """
        Fitting Function for Population Decay Curve
        """
        return a * np.exp(- t / T1) + b

    def _guess_init_params(self) -> None:
        a0 = self.signal[0] - self.signal[-1]
        b0 = self.signal[-1]

        mid_idx = np.argmin(np.abs(self.signal - (a0 / 2 + b0)))

        T10 = (
            (self.time[0] - self.time[mid_idx]) /
            np.log(1 - (self.signal[0] - self.signal[mid_idx]) / a0)
        )

        self.p0 = [T10, a0, b0]
        # lb = [-np.inf, 0, np.min(self.signal)]
        # ub = [np.inf, (np.max(self.signal) - np.min(self.signal)),
        #       np.max(self.signal)]

    def _save_fit_results(self, popt: np.ndarray, pcov: np.ndarray) -> None:
        super()._save_fit_results(popt, pcov)
        self.T1 = popt[0]
        self.T1_sigma_err = np.sqrt(pcov[0, 0])

    def plot_result(self, fit_n_pts: int = 1000) -> plt.Figure:
        fig = super().plot_result(fit_n_pts = fit_n_pts)
        # add title showing the fit result
        T1_unitful = UnitfulNumber(
            self.T1, error = self.T1_sigma_err, base_unit = self.time_unit
        )
        plt.title(f"$T_1$ = {T1_unitful}")
        plt.legend(fontsize='x-small')
        fig.tight_layout()
        plt.show()
        return fig


class Ramsey(TimeSweep):
    """
    Class to perform analysis and visualize Ramsey fringes data.
    """
    def fit_func(self, t: np.ndarray, *args) -> np.ndarray:
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

    def analyze(
        self, p0: list[float] = None, plot: bool = True,
        fit_gaussian: bool = False, **kwargs
    ) -> None:
        """
        Analyze the data with initial parameter `p0`.
        """
        self.fit_gaussian = fit_gaussian
        super().analyze(p0=p0, plot=plot, **kwargs)

    def _guess_init_params(self) -> None:
        b0: float = np.mean(self.signal)
        signal0: float = self.signal - b0
        amax: float = np.max(np.abs(signal0))

        # perform fft to find frequency of Ramsey fringes
        freq = np.fft.rfftfreq(
            len(self.signal), d = (self.time[1] - self.time[0])
        )
        δf0 = freq[np.argmax(np.abs(np.fft.rfft(self.signal - np.mean(self.signal))))]
        df = freq[1] - freq[0] # frequency step of FFT

        # in-phase and quadrature envelope
        envComplex = get_envelope(signal0, self.time, δf0)
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

    def _save_fit_results(self, popt: np.ndarray, pcov: np.ndarray):
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

    def plot_result(self, fit_n_pts: int = 1000) -> plt.Figure:
        fig = super().plot_result()

        if not self.fit_gaussian:
            T2Ramsey_unitful = UnitfulNumber(
                self.T2Ramsey, error = self.T2Ramsey_sigma_err,
                base_unit = self.time_unit
            )
            T2_string = r"$T_2^*$ = " + f"{T2Ramsey_unitful}"
        else:
            Texp_unitful = UnitfulNumber(
                self.Texp, self.Texp_sigma_err, base_unit = self.time_unit
            )
            Texp_string = r"$T_\mathrm{exp}$ = " + f"{Texp_unitful}"

            Tgauss_unitful = UnitfulNumber(
                self.Tgauss, error = self.Tgauss_sigma_err,
                base_unit = self.time_unit
            )
            Tgauss_string = r"$T_\mathrm{gauss}$ = " + "{Tgauss_unitful}"
            
            T2_string = ', '.join([Texp_string, Tgauss_string])

        delta_freq_unitful = UnitfulNumber(
            self.delta_freq, error = self.Tgauss_sigma_err, base_unit = 'Hz'
        )
        delta_freq_string = r"$\Delta f$ = " + f"{delta_freq_unitful}"

        plt.title(', '.join([T2_string, delta_freq_string]), fontsize='small')
        plt.legend(loc=0, fontsize='x-small')
        fig.tight_layout()
        plt.show()
        return fig


class HahnEcho(TimeSweep):
    """
    Class to analyze and visualize Hahn echo decay (T2 Hahn echo experiment) data.
    """
    def fit_func(self, t: np.ndarray, *args) -> np.ndarray:
        """
        Fitting Function for Hahn Echo. If `fit_gaussian` is False (default)
        """
        if not self.fit_gaussian:
            T2, a, b = args
            return a * np.exp(- t / T2) + b
        else:
            Texp, Tgauss, a, b = args
            return a * np.exp(- t / Texp - (t / Tgauss) ** 2) + b

    def analyze(
        self, p0: list[float] = None, plot: bool = True,
        fit_gaussian: bool = False, **kwargs
    ):
        """
        Analyze the data with initial parameter `p0`.
        """
        self.fit_gaussian = fit_gaussian
        super().analyze(p0 = p0, plot = plot, **kwargs)

    def _guess_init_params(self) -> None:
        a0 = self.signal[0] - self.signal[-1]
        b0 = self.signal[-1]

        mid_idx = np.argmin(np.abs(self.signal - (a0 / 2 + b0)))
        T20 = (
            (self.time[0] - self.time[mid_idx]) /
            np.log(1 - (self.signal[0] - self.signal[mid_idx]) / a0)
        )

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

    def _save_fit_results(self, popt: np.ndarray, pcov: np.ndarray):
        super()._save_fit_results(popt, pcov)
        if not self.fit_gaussian:
            self.T2Echo = popt[0]
            self.T2Echo_sigma_err = np.sqrt(pcov[0, 0])
        else:
            self.Texp = popt[0]
            self.Texp_sigma_err = np.sqrt(pcov[0, 0])
            self.Tgauss = popt[1]
            self.Tgauss_sigma_err = np.sqrt(pcov[1, 1])

    def plot_result(self, fit_n_pts: int = 1000) -> plt.Figure:
        fig = super().plot_result(fit_n_pts = fit_n_pts)

        if not self.fit_gaussian:
            T2Echo_unitful = UnitfulNumber(
                self.T2Echo, error = self.T2Echo_sigma_err, base_unit = self.time_unit
            )
            T2_string = r"$T_{2E}$ = " + f"{T2Echo_unitful}"
        else:
            Texp_unitful = UnitfulNumber(
                self.Texp, error = self.Texp_sigma_err, base_unit = self.time_unit
            )
            Texp_string = r"$T_\mathrm{exp}$ = " + f"{Texp_unitful}"
            Tgauss_unitful = UnitfulNumber(
                self.Tgauss, error = self.Tgauss_sigma_err, base_unit = self.time_unit
            )
            Tgauss_string = r"$T_\mathrm{gauss}$ = " + f"{Tgauss_unitful}"
            T2_string = ', '.join([Texp_string, Tgauss_string])
        plt.title(T2_string, fontsize='small')
        fig.tight_layout()
        plt.show()
        return fig


class PowerRabi(AmpSweep):

    def fit_func(self, amp: np.ndarray, amp_pi: float, a: float, b: float) -> np.ndarray:
        """
        Rabi oscillation curve with fixed pulse duration and only driving amplitude swept
        a / 2 (1 - cos(π * amp / amp_pi)) + b
        Here amp_pi is the amplitude corresponding to the pi-pulse.
        """
        return a / 2 * (1 - np.cos(np.pi * amp / amp_pi)) + b

    def _guess_init_params(self):
        # perform fft to find frequency of Rabi oscillation
        freq = np.fft.rfftfreq(
            len(self.signal), d=(self.amp[1] - self.amp[0])
        )

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

    def _save_fit_results(self, popt: np.ndarray, pcov: np.ndarray):
        super()._save_fit_results(popt, pcov)
        self.amp_pi = popt[0]
        self.amp_pi2 = self.amp_pi / 2
        self.amp_pi_sigma_err = np.sqrt(pcov[0, 0])
        self.amp_pi2_sigma_err = np.sqrt(pcov[0, 0]) / 2

    def plot_result(self, fit_n_pts: int = 1000) -> plt.Figure:
        fig = super().plot_result(fit_n_pts = fit_n_pts)

        amp_pi_unitful = UnitfulNumber(
            self.amp_pi, error = self.amp_pi_sigma_err, base_unit = self.amp_unit
        )
        plt.title(r"$a_{\pi} = " + f"{amp_pi_unitful}")
        plt.legend(loc=0, fontsize='x-small')
        fig.tight_layout()
        plt.show()
        return fig
