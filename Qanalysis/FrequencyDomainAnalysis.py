# Written by Eunjong Kim

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class FrequencyDomainAnalysis:
    """
    Class for implementing fitting of lineshape in various configurations
    """
    def __init__(self, freq, data, df=0, fit_mag_dB=False, plot_mag_dB=False):
        self.frequency = freq
        self.data = data
        self.fit_type = None
        self.fit_result = None
        self.fit_mag_dB = fit_mag_dB
        self.plot_mag_dB = plot_mag_dB
        self.is_analyzed = False

        # initialize fit parameters
        self.p0 = None
        self.p0_mag, self.p0_ang = self._init_fit_params(df)
        
        
        self.p1 = None
        self.results = None
        self.solver_options = {'maxiter': 100000, 'maxfev': 100000, 'xatol': 1e-10, 'fatol': 1e-10}
    
    def _init_fit_params(self, df):
        # to be overwritten in subclasses
        pass

    def _estimate_f0_FWHM(self):
        f = self.frequency
        mag2 = np.abs(self.data) ** 2
        
        f0 = f[mag2.argmin()]
        smin, smax = np.min(mag2), np.max(mag2)
        
        # data in frequency < f0 or frequency >= f0
        f_l, s_l = f[f < f0], mag2[f < f0]
        f_r, s_r = f[f >= f0], mag2[f >= f0]
        
        f1 = f_l[np.abs(s_l - 0.5 * (smin + smax)).argmin()]
        f2 = f_r[np.abs(s_r - 0.5 * (smin + smax)).argmin()]
        
        # numerically find full width half max from magnitude squared
        Δf = f2 - f1
        
        return f0, Δf

    def _plot_base(self):
        fig = plt.figure(figsize=(10, 4))
        grid = plt.GridSpec(2, 2, wspace=0.6, hspace=0.3)
        main_ax = fig.add_subplot(grid[:, 1])
        mag_ax = fig.add_subplot(grid[0, 0])
        ang_ax = fig.add_subplot(grid[1, 0])

        main_ax.scatter(self.data.real, self.data.imag, s=2, label="Data", color="blue")
        window = 1.05 * np.max(np.abs(self.data))
        
        main_ax.set_xlim([- window, window])
        main_ax.set_ylim([- window, window])

        main_ax.set_xlabel("Real", fontsize=14)
        main_ax.set_ylabel("Imag", fontsize=14)
        main_ax.axvline(0, color="black", lw=1)
        main_ax.axhline(0, color="black", lw=1)
        # plot mag axis
        if self.plot_mag_dB:
            mag_ax.scatter(self.frequency/1e9,
                           self._dB(np.abs(self.data) ** 2), s=2, color="blue")
        else:
            mag_ax.scatter(self.frequency/1e9, np.abs(self.data), s=2, color="blue")
            
        mag_ax.set_ylabel(r"Magnitude" + self.plot_mag_dB * " (dB)", fontsize=12)

        # plot ang axis
        ang_ax.scatter(self.frequency/1e9, np.unwrap(np.angle(self.data)), s=2, color="blue")
        ang_ax.set_xlabel("Frequency (GHz)", fontsize=12)
        ang_ax.set_ylabel(r"Phase (rad)", fontsize=12)
        axes = [main_ax, mag_ax, ang_ax]
        return fig, axes

    def fit_func(self):
        # to be overwritten in subclasses
        pass
    
    def _dB(self, x):
        return 10 * np.log10(x)

    def analyze_mag_dB(self):
        """
        Analyze the data by fitting only on magnitude with dB scale. The optimization
        on phase fit parameters is not perfomed in this case.
        """
        # fit function with two separate parameter sets (mag, ang)
        _mag_ang_func = lambda f, p_mag, p_ang: self.fit_func(f, *(list(p_mag) + list(p_ang)))
        p0_mag, p0_ang = self.p0_mag, self.p0_ang
        
        lsqfun_mag_dB = lambda p: \
            np.sqrt(np.sum((self._dB(np.abs(_mag_ang_func(self.frequency, p, p0_ang)) ** 2) - self._dB(abs(self.data) ** 2)) ** 2))
        
        result_mag_dB = minimize(lsqfun_mag_dB, p0_mag, method='Nelder-Mead', options=self.solver_options)

        self.p1 = list(result_mag_dB.x) + list(p0_ang)

        Q = self.p1[1] * 1e3
        Qe = Q / self.p1[2]
        Qi = 1 / (1 / Q - 1 / Qe)
        f0 = self.p1[0] * 1e9
        self.results = {'f0': f0, 'Qe': Qe, 'Qi': Qi}
        self.is_analyzed = True
        return self.results

    def analyze(self):
        """
        Analyze the data using a complex fitting function simultaneously fitting
        real and imaginary part of the data
        """
        # fit function with two separate parameter sets (mag, ang)
        _mag_ang_func = lambda f, p_mag, p_ang: self.fit_func(f, *(list(p_mag) + list(p_ang)))
        p0_mag, p0_ang = self.p0_mag, self.p0_ang

        # Fit Magnitude first
        lsqfun_mag = lambda p: np.sqrt(np.sum((np.abs(_mag_ang_func(self.frequency, p, p0_ang)) - abs(self.data)) ** 2))
        if self.fit_mag_dB:
            lsqfun_mag = lambda p: np.sqrt(np.sum((self._dB(np.abs(_mag_ang_func(self.frequency, p, p0_ang))) - self._dB(abs(self.data))) ** 2))

        
        result_mag = minimize(lsqfun_mag, p0_mag, method='Nelder-Mead', options=self.solver_options)
        
        p1_mag = result_mag.x
        # Two different fits on angle, to take into account the possibility that
        # the extracted parameter `QoverQe` was actually Q/Qi instead of Q/Qe.
        # This is captured by two different lsqfits on angles utilizing the
        # relation 1 = Q/Qe + Q/Qi.
        p1_mag = [p1_mag[0], abs(p1_mag[1]), p1_mag[2], p1_mag[3], p1_mag[4]]
        p1_mag1 = [p1_mag[0], abs(p1_mag[1]), (1 - p1_mag[2]), p1_mag[3], p1_mag[4]]
        print(p1_mag, p1_mag1)
        lsqfun_ang = lambda p: \
            np.sqrt(np.sum(np.abs(np.angle(_mag_ang_func(self.frequency, p1_mag, p)) - np.angle(self.data)) ** 2))
        lsqfun_ang1 = lambda p: np.sqrt(np.sum(np.abs(np.angle(_mag_ang_func(self.frequency, p1_mag1, p)) - np.angle(self.data)) ** 2))

        result_ang = minimize(lsqfun_ang, p0_ang, method='Nelder-Mead', options=self.solver_options)
        p1_ang = result_ang.x

        result_ang1 = minimize(lsqfun_ang1, p0_ang, method='Nelder-Mead', options=self.solver_options)
        p1_ang1 = result_ang1.x
        
        # choose the fit that minimizes the lsq function
        if result_ang1.fun < result_ang.fun:
            p1_mag = p1_mag1
            p1_ang = p1_ang1

        # Finally, perform simultaneous fit using the initial parameters from fits above.
        lsqfun = lambda p: np.sqrt(np.sum(np.abs(self.fit_func(self.frequency, *p) - self.data) ** 2))
        result = minimize(lsqfun, list(p1_mag) + list(p1_ang), method='Nelder-Mead', options=self.solver_options)
        
        self.p1 = result.x

        Q = self.p1[1] * 1e3
        Qe = Q / self.p1[2]
        Qi = 1 / (1 / Q - 1 / Qe)
        f0 = self.p1[0] * 1e9
        self.results = {'f0': f0, 'Qe': Qe, 'Qi': Qi}
        self.is_analyzed = True
        return self.results

    def plot_result(self):
        """
        Will be overwritten in subclass
        """
        if not self.is_analyzed:
            raise ValueError("The data must be analyzed before plotting")

class SingleSidedS11Fit(FrequencyDomainAnalysis):
    """
    Class for imlementing fitting of lineshape in reflection measurement
    """
    def __init__(self, freq, data, df=0, fit_mag_dB=False, plot_mag_dB=False):
        super().__init__(freq, data, df=df, fit_mag_dB=fit_mag_dB, plot_mag_dB=plot_mag_dB)
        
        self.fit_type = "SingleSidedReflection"

        
        
    def fit_func(self, f, f0, Q, QoverQe, δf, a0, ϕ0, τ0):
        """
        Reflection fit for resonator single-sided coupled to waveguide
        according to relation
        S11 = a0 * exp(i(ϕ0 + 2pi * (f - f[0]) * τ0)) .* ...
             (1 - 2 * QoverQe * (1 + 2i * δf / f0) / (1 + 2i * Q * (f - f0) / f0))
             
        Note: If df = 0 (no background) this gives
        S11 = 1 - (2 kappa_e) / (kappa_i + kappa_e + 2i (omega - omega0)) 
            = (kappa_i - kappa_e + 2i (omega - omega0)) / (kappa_i + kappa_e + 2i (omega - omega0))
        See Aspelmeyer et al, "Cavity Optomechanics", Rev. Mod. Phys. 86, 1391 (2014).
        """
        
        return (a0 * np.exp(1j * (ϕ0 + 2 * np.pi * (f - f[0]) /1e9 * τ0)) *
                (1 - 2 * QoverQe * (1 + 2j * δf / f0) /
                (1 + 2j * (Q * 1e3) * (f - f0 * 1e9) / (f0 * 1e9))))

    
    def plot_result(self):
        # get most of the plotting done
        fig, axes = self._plot_base()
        
        freq_fit = np.linspace(self.frequency[0], self.frequency[-1], 500)
        _fit0 = self.fit_func(freq_fit, *(self.p0))
        _fit1 = self.fit_func(freq_fit, *(self.p1))
        
        main_ax, mag_ax, ang_ax = axes
        
        main_ax.plot(_fit0.real, _fit0.imag, ls=":", color="orange", label="Initial Params")
        main_ax.plot(_fit1.real, _fit1.imag, ls="-", color="black", label="Fit")
        main_ax.legend()

        if self.plot_mag_dB:
            mag_ax.plot(freq_fit/1e9, self._dB(np.abs(_fit0) ** 2),
                        ls=':', color='orange')
            mag_ax.plot(freq_fit/1e9, self._dB(np.abs(_fit1) ** 2),
                        ls='-', color='black')            
        else:
            mag_ax.plot(freq_fit/1e9, np.abs(_fit0), ls=':', color='orange')
            mag_ax.plot(freq_fit/1e9, np.abs(_fit1), ls='-', color='black')

        ang_ax.plot(freq_fit/1e9, np.unwrap(np.angle(_fit0)), ls=':', color='orange')
        ang_ax.plot(freq_fit/1e9, np.unwrap(np.angle(_fit1)), ls='-', color='black')

        results = self.results
        f0, Qe, Qi = results['f0'], results['Qe'], results['Qi']
        κe, κi = 2 * np.pi * f0/Qe, 2 * np.pi * f0/Qi
        mag_ax.set_title(r"$f_0 = %.3f$ GHz, $\kappa_e/2\pi = %.3f$ MHz, $\kappa_i/2\pi = %.3f$ MHz" % (f0/1e9, κe/(2e6 * np.pi), κi/(2e6 * np.pi)))
        main_ax.set_title(r"$Q_e = %d$, $Q_i = %d$" % (Qe, Qi))

    def _init_fit_params(self, df):
        # magnitude data
        _mag = np.abs(self.data)
        # phase data
        _ang = np.angle(self.data)
        # unwrapped phase data
        _angU = np.unwrap(_ang)
        
        f = self.frequency
        
        a0 = _mag[0]
        ϕ0 = _angU[0]
        τ0 = 0.0
        if np.abs(_angU[-1] - _angU[0]) > 2.1 * np.pi:
            # if phase data at start and stop frequencies differ more than 2pi,
            # perform phase subtraction associated with delay
            τ0 = (_angU[-1] - _angU[0]) / ((f[-1] - f[0]) / 1e9)/ (2 * np.pi)

        # Estimate total Q from the FWHM in |mag|^2
        f0, Δf = self._estimate_f0_FWHM()
        QoverQe0 = 0.5 * (1 - np.min(_mag) / a0)
        Q0 = f0 / Δf / 1e3 # Q in units of 1000
        f0 /= 1e9 # f0 in units of GHz
        
        p0_mag = [f0, Q0, QoverQe0, df, a0]
        p0_ang = [ϕ0, τ0]
        self.p0 = p0_mag + p0_ang
        return p0_mag, p0_ang
        

class WaveguideCoupledS21Fit(FrequencyDomainAnalysis):
    """
    Class for imlementing fitting of lineshape in reflection measurement
    """
    def __init__(self, freq, data, df=0, fit_mag_dB=False, plot_mag_dB=False):
        super().__init__(freq, data, df=df, fit_mag_dB=fit_mag_dB, plot_mag_dB=plot_mag_dB)
        
        self.fit_type = "WaveguideCoupledTransmission"


        
    def fit_func(self, f, f0, Q, QoverQe, δf, a0, ϕ0, τ0):
        """
        Reflection fit for resonator single-sided coupled to waveguide
        according to relation
        S11 = a0 * exp(i(ϕ0 + 2pi * (f - f[0]) * τ0)) .* ...
             (1 - QoverQe * (1 + 2i * δf / f0) / (1 + 2i * Q * (f - f0) / f0))
             
        Note: If df = 0 (no background) this gives
        S21 = 1 - (kappa_e) / (kappa_i + kappa_e + 2i (omega - omega0))
            = (kappa_i + 2i (omega - omega0)) / (kappa_i + kappa_e + 2i (omega - omega0))
        See Khalil et al, "An analysis method for asymmetric resonator
        transmission applied to superconducting devices",
        J. Appl. Phys. 111, 054510 (2012).
        """
        
        return (a0 * np.exp(1j * (ϕ0 + 2 * np.pi * (f - f[0]) /1e9 * τ0)) *
                (1 - QoverQe * (1 + 2j * δf / f0) /
                (1 + 2j * (Q * 1e3) * (f - f0 * 1e9) / (f0 * 1e9))))

    
    def plot_result(self):
        # get most of the plotting done
        fig, axes = self._plot_base()
        
        freq_fit = np.linspace(self.frequency[0], self.frequency[-1], 500)
        _fit0 = self.fit_func(freq_fit, *(self.p0))
        _fit1 = self.fit_func(freq_fit, *(self.p1))
        
        main_ax, mag_ax, ang_ax = axes
        
        main_ax.plot(_fit0.real, _fit0.imag, ls=":", color="orange", label="Initial Params")
        main_ax.plot(_fit1.real, _fit1.imag, ls="-", color="black", label="Fit")
        main_ax.legend()
        
        if self.plot_mag_dB:
            mag_ax.plot(freq_fit/1e9, self._dB(np.abs(_fit0) ** 2),
                        ls=':', color='orange')
            mag_ax.plot(freq_fit/1e9, self._dB(np.abs(_fit1) ** 2),
                        ls='-', color='black')            
        else:
            mag_ax.plot(freq_fit/1e9, np.abs(_fit0), ls=':', color='orange')
            mag_ax.plot(freq_fit/1e9, np.abs(_fit1), ls='-', color='black')

        ang_ax.plot(freq_fit/1e9, np.unwrap(np.angle(_fit0)), ls=':', color='orange')
        ang_ax.plot(freq_fit/1e9, np.unwrap(np.angle(_fit1)), ls='-', color='black')
       
        results = self.results
        f0, Qe, Qi = results['f0'], results['Qe'], results['Qi']
        κe, κi = 2 * np.pi * f0/Qe, 2 * np.pi * f0/Qi
        mag_ax.set_title(r"$f_0 = %.3f$ GHz, $\kappa_e/2\pi = %.3f$ MHz, $\kappa_i/2\pi = %.3f$ MHz" % (f0/1e9, κe/(2e6 * np.pi), κi/(2e6 * np.pi)))
        main_ax.set_title(r"$Q_e = %d$, $Q_i = %d$" % (Qe, Qi))

    def _init_fit_params(self, df):
        # magnitude data
        _mag = np.abs(self.data)
        # phase data
        _ang = np.angle(self.data)
        # unwrapped phase data
        _angU = np.unwrap(_ang)
        
        f = self.frequency
        
        a0 = _mag[0]
        ϕ0 = _angU[0]
        τ0 = (_angU[-1] - _angU[0]) / ((f[-1] - f[0]) / 1e9) / (2 * np.pi)

        # Estimate total Q from the FWHM in |mag|^2
        f0, Δf = self._estimate_f0_FWHM()
        QoverQe0 = (1 - np.min(_mag) / a0)
        Q0 = f0 / Δf / 1e3 # Q in units of 1000
        f0 /= 1e9 # f0 in units of GHz
        
        p0_mag = [f0, Q0, QoverQe0, df, a0]
        p0_ang = [ϕ0, τ0]
        self.p0 = p0_mag + p0_ang
        return p0_mag, p0_ang
