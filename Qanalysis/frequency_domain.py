# Written by Eunjong Kim

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from lmfit import Parameters, Minimizer, report_fit, report_ci, conf_interval
from collections import OrderedDict

class FrequencyDomain:
    """
    Class for implementing fitting of lineshape in various configurations
    """
    def __init__(self, freq, data, df=0, fit_mag_dB=False, plot_mag_dB=False):
        self.frequency = freq
        self.data = data
        self.fit_type = None
        self.result = None
        self.fit_mag_dB = fit_mag_dB
        self.plot_mag_dB = plot_mag_dB
        self.is_analyzed = False
        self.mini = None
        # initialize fit parameters
        self.p0 = None
        self.p0_mag, self.p0_ang = self._init_fit_params(df)

        self.ci = None
        self.n_sigma = None
        self.__plot_window = None
        self.p1 = None
        self.__solver_options = {'maxiter': 100000, 'maxfev': 100000, 'xatol': 1e-10, 'fatol': 1e-10}

    def _init_fit_params(self, df):
        # to be overwritten in subclasses
        pass

    def _prepare_fit_params(self, f0, Q0, QoverQe0, df, a0, ϕ0, τ0):

        f = self.frequency
        kappa_eOver2pi0 = f0 / (Q0 / QoverQe0)
        kappa_iOver2pi0 = f0 / Q0 - kappa_eOver2pi0

        # initialize magnitude fit parameters
        p0_mag = Parameters()
        p0_mag.add('f0_MHz', value=f0 / 1e6,
                   min=np.min(f / 1e6),
                   max=np.max(f / 1e6))
        p0_mag.add('kappa_eOver2pi_MHz', value=kappa_eOver2pi0 / 1e6,
                   min=0),
                   max=(np.max(f) - np.min(f)) / 1e6)
        p0_mag.add('kappa_iOver2pi_MHz', value=kappa_iOver2pi0 / 1e6,
                   min=0),
                   max=(np.max(f) - np.min(f)) / 1e6)
        p0_mag.add('kappaOver2pi_MHz',
                   expr='kappa_eOver2pi_MHz + kappa_iOver2pi_MHz',
                   min=0,
                   max=(np.max(f) - np.min(f)) / 1e6)

        p0_mag.add('Q', expr='f0_MHz / kappaOver2pi_MHz',
                   min=(f0 / 1e6) / ((np.max(f) - np.min(f)) / 1e6))
        p0_mag.add('Qe', expr='f0_MHz / kappa_eOver2pi_MHz',
                   min=(f0 / 1e6) / ((np.max(f) - np.min(f)) / 1e6))
        p0_mag.add('QoverQe', expr='Q / Qe', max=1, min=0)
        p0_mag.add('Qi', expr='f0_MHz / kappa_iOver2pi_MHz', min=0)
        p0_mag.add('df', value=df)
        p0_mag.add('A', value=a0, min=0, max=1.5)


        # initialize angle fit parameters
        p0_ang = Parameters()
        p0_ang.add('phi', value=ϕ0, min=-np.pi, max=np.pi)
        p0_ang.add('tau_ns', value=τ0 / 1e-9)
        return p0_mag, p0_ang

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
        self.__plot_window = window
        main_ax.set_xlabel("Real", fontsize=14)
        main_ax.set_ylabel("Imag", fontsize=14)
        main_ax.axvline(0, color="black", lw=1, alpha=0.2)
        main_ax.axhline(0, color="black", lw=1, alpha=0.2)

        # plot mag axis
        if self.plot_mag_dB:
            mag_ax.scatter(self.frequency/1e9,
                           self._dB(np.abs(self.data) ** 2),
                           s=2, color="blue")
        else:
            mag_ax.scatter(self.frequency/1e9, np.abs(self.data),
                           s=2, color="blue")

        mag_ax.set_ylabel(r"Magnitude" + self.plot_mag_dB * " (dB)",
                          fontsize=12)

        # plot ang axis
        ang_ax.scatter(self.frequency/1e9, np.unwrap(np.angle(self.data)),
                       s=2, color="blue")
        ang_ax.set_xlabel("Frequency (GHz)", fontsize=12)
        ang_ax.set_ylabel(r"Phase (rad)", fontsize=12)

        min_freq, max_freq = np.min(self.frequency), np.max(self.frequency)
        mag_ax.set_xlim([min_freq / 1e9, max_freq / 1e9])
        ang_ax.set_xlim([min_freq / 1e9, max_freq / 1e9])

        axes = [main_ax, mag_ax, ang_ax]
        return fig, axes

    def fit_func(self):
        # to be overwritten in subclasses
        pass

    def _dB(self, x):
        return 10 * np.log10(x)

    def _mag_params_to_array(self, params):
        """
        Put lmfit.Parameters to numpy array
        """
        f0_MHz = params['f0_MHz']
        Q = params['Q']
        QoverQe = params['QoverQe']
        δf = params['df']
        a = params['A']

        p_mag = [f0_MHz, Q, QoverQe, δf, a]
        return p_mag

    def _ang_params_to_array(self, params):
        """
        Put lmfit.Parameters to numpy array
        """
        ϕ = params['phi']
        τ_ns = params['tau_ns']

        p_ang = [ϕ, τ_ns]
        return p_ang

    def _params_to_array(self, params):
        """
        Put lmfit.Parameters to numpy array
        """
        p_mag = self._mag_params_to_array(params)
        p_ang = self._ang_params_to_array(params)

        return list(p_mag) + list(p_ang)

    def _conf_interval(self, advanced_ci, n_sigma):
        if advanced_ci:
            return conf_interval(self.mini, self.fit_result, sigmas=[n_sigma])
        else:
            ci = OrderedDict()
            params = self.fit_result.params
            for p in params:
                name = p
                sigma = params[p].stderr

                mu = params[p].value
                prob = self._n_sigma_to_prob(n_sigma)

                _min, _max = (mu - n_sigma * sigma), (mu + n_sigma * sigma)
                ci[name] = [(prob, _min),
                            (0.0, mu),
                            (prob, _max)]
            return ci

    def _n_sigma_to_prob(self, n_sigma):
        """
        Convert number of sigmas to probablility
        """
        return erf(n_sigma/np.sqrt(2))

    def analyze_mag_dB(self, n_sigma=2, report=True, advanced_ci=False):
        """
        Analyze the data by fitting only on magnitude with dB scale. The optimization
        on phase fit parameters is not perfomed in this case.
        """
        # fit function with two separate parameter sets (mag, ang)
        p0_mag, p0_ang = self.p0_mag, self.p0_ang

        def _mag_dB_minfunc(mag_params, freq, data):
            """
            Residual function for dB magnitude fit
            """
            p_mag = self._mag_params_to_array(mag_params)
            p_ang0 = self._ang_params_to_array(p0_ang)
#            return np.abs(self._dB(np.abs(self.fit_func(freq, *(list(p_mag)+list(p_ang0)))) ** 2) -
#                          self._dB(np.abs(data) ** 2))
            return np.abs(self._dB(np.abs(self.fit_func(freq, *(list(p_mag)+list(p_ang0)))) ** 2) -
                          self._dB(np.abs(data) ** 2))

        self.mini = Minimizer(_mag_dB_minfunc, p0_mag,
                              fcn_args=(self.frequency, self.data), calc_covar=True)
        self.fit_result = self.mini.minimize(method='nelder',
                                             options=self.__solver_options)

        self.p1 = self.fit_result.params + p0_ang
        self.is_analyzed = True

        f0 = self.p1['f0_MHz'] * 1e6
        kappa_eOver2pi = self.p1['kappa_eOver2pi_MHz'] * 1e6
        kappa_iOver2pi = self.p1['kappa_iOver2pi_MHz'] * 1e6
        Q = self.p1['Q']
        Qe = self.p1['Qe']
        Qi = self.p1['Qi']

        self.n_sigma = n_sigma
        # calculate confidence interval


        self.ci = self._conf_interval(advanced_ci, n_sigma)

        prob_n = self._n_sigma_to_prob(n_sigma)
        bound = {}
        for name in ['f0_MHz', 'kappa_eOver2pi_MHz', 'kappa_iOver2pi_MHz']:
            _ci = self.ci[name]
            ci_nsigma = np.array([c[1] for c in _ci if np.abs(c[0] - prob_n) < 1e-5])

            lowerbound = np.min(ci_nsigma)
            upperbound = np.max(ci_nsigma)

            bound[name] = np.array([lowerbound, upperbound])

        if report:
            # report fit
            report_fit(self.fit_result)

            if advanced_ci:
                # report confidence_interval
                report_ci(self.ci)

        self.results = {'f0': (f0, 1e6 * bound['f0_MHz']),
                        'kappa_eOver2pi': (kappa_eOver2pi,
                                           1e6 * bound['kappa_eOver2pi_MHz']),
                        'kappa_iOver2pi': (kappa_iOver2pi,
                                           1e6 * bound['kappa_iOver2pi_MHz']),
                        'Qe': (Qe, ),
                        'Qi': (Qi, )}
        self.is_analyzed = True
        return self.results
        return result

    def analyze(self, n_sigma=2, report=True, advanced_ci=False):
        """
        Analyze the data using a complex fitting function simultaneously fitting
        real and imaginary part of the data
        """
        # fit function with two separate parameter sets (mag, ang)
        _mag_ang_func = lambda f, p_mag, p_ang: self.fit_func(f, *(list(p_mag) + list(p_ang)))
        p0_mag, p0_ang = self.p0_mag, self.p0_ang

        # Fit Magnitude first
        def _mag_minfunc(mag_params, ang_params, freq, data):
            """
            Residual function for magnitude fit
            """
            p_mag = self._mag_params_to_array(mag_params)
            p_ang0 = self._ang_params_to_array(ang_params)
            return np.abs(np.abs(self.fit_func(freq, *(list(p_mag)+list(p_ang0)))) -
                          np.abs(data))

        def _mag_dB_minfunc(mag_params, ang_params, freq, data):
            """
            Residual function for dB magnitude fit
            """
            p_mag = self._mag_params_to_array(mag_params)
            p_ang0 = self._ang_params_to_array(ang_params)
            return np.abs(self._dB(np.abs(self.fit_func(freq, *(list(p_mag)+list(p_ang0)))) ** 2) -
                          self._dB(np.abs(data) ** 2))

        mini_mag = Minimizer(_mag_minfunc, p0_mag,
                             fcn_args=(p0_ang, self.frequency, self.data))


        result_mag = mini_mag.minimize(method='nelder',
                                       options=self.__solver_options)

        p1_mag = result_mag.params.copy()
        p1_mag1 = result_mag.params.copy()

        # swap kappa_e and kappa_i for p1_mag1 to consider the case were kappa_i
        # and kappa_e are swapped
        k1, k2 = p1_mag1['kappa_eOver2pi_MHz'].value, p1_mag1['kappa_iOver2pi_MHz'].value
        p1_mag1['kappa_eOver2pi_MHz'].value = k2
        p1_mag1['kappa_iOver2pi_MHz'].value = k1

        def _ang_minfunc(ang_params, mag_params, freq, data):
            """
            Residual function for angle fit
            """
            p_mag0 = self._mag_params_to_array(mag_params)
            p_ang = self._ang_params_to_array(ang_params)
            return np.abs(np.angle(self.fit_func(freq, *(list(p_mag0)+list(p_ang)))) -
                          np.angle(data))

        mini_ang = Minimizer(_ang_minfunc, p0_ang,
                             fcn_args=(p1_mag, self.frequency, self.data))
        result_ang = mini_ang.minimize(method='nelder',
                                       options=self.__solver_options)

        mini_ang1 = Minimizer(_ang_minfunc, p0_ang,
                              fcn_args=(p1_mag1, self.frequency, self.data))
        result_ang1 = mini_ang1.minimize(method='nelder',
                                         options=self.__solver_options)

        # Two different fits on angle, to take into account the possibility that
        # the extracted parameter `kappa_eOver2pi_MHz` was actually 'kappa_eOver2pi_MHz'e.
        # This is captured by two different lsqfits on angles utilizing the
        # relation 1 = Q/Qe + Q/Qi.
        p1_ang = result_ang.params.copy()
        p1_ang1 = result_ang1.params.copy()

        # choose the fit that minimizes the lsq function
        if result_ang1.chisqr < result_ang.chisqr:
            p1_mag = p1_mag1
            p1_ang = p1_ang1

        def _simultaneous_minfunc(params, freq, data):
            """
            Residual function for magnitude fit
            """
            p0 = self._params_to_array(params)
            return np.abs(self.fit_func(freq, *p0) - (data))
        # Finally, perform simultaneous fit using the initial parameters from fits above.
        self.mini = Minimizer(_simultaneous_minfunc, p1_mag + p1_ang,
                              fcn_args=(self.frequency, self.data))
        self.fit_result = self.mini.minimize(method='nelder')

        self.p1 = self.fit_result.params

        f0 = self.p1['f0_MHz'] * 1e6
        kappa_eOver2pi = self.p1['kappa_eOver2pi_MHz'] * 1e6
        kappa_iOver2pi = self.p1['kappa_iOver2pi_MHz'] * 1e6
        Q = self.p1['Q']
        Qe = self.p1['Qe']
        Qi = self.p1['Qi']

        self.n_sigma = n_sigma
        self.ci = self._conf_interval(advanced_ci, n_sigma)

        prob_n = self._n_sigma_to_prob(n_sigma)
        bound = {}
        for name in ['f0_MHz', 'kappa_eOver2pi_MHz', 'kappa_iOver2pi_MHz']:
            _ci = self.ci[name]
            ci_nsigma = np.array([c[1] for c in _ci if np.abs(c[0] - prob_n) < 1e-5])

            lowerbound = np.min(ci_nsigma)
            upperbound = np.max(ci_nsigma)

            bound[name] = np.array([lowerbound, upperbound])

        if report:
            # report fit
            report_fit(self.fit_result)

            if advanced_ci:
                # report confidence_interval
                report_ci(self.ci)

        self.results = {'f0': (f0, 1e6 * bound['f0_MHz']),
                        'kappa_eOver2pi': (kappa_eOver2pi,
                                           1e6 * bound['kappa_eOver2pi_MHz']),
                        'kappa_iOver2pi': (kappa_iOver2pi,
                                           1e6 * bound['kappa_iOver2pi_MHz']),
                        'Qe': (Qe, ),
                        'Qi': (Qi, )}
        self.is_analyzed = True
        return self.results
        return result

    def plot_result(self):
        """
        Plot the result of fitting together with parameter confidence interval.
        """
        if not self.is_analyzed:
            raise ValueError("The data must be analyzed before plotting")

        # get most of the plotting done
        fig, axes = self._plot_base()

        freq_fit = np.linspace(self.frequency[0], self.frequency[-1], 500)
        _fit0 = self.fit_func(freq_fit, *(self._params_to_array(self.p0)))
        _fit1 = self.fit_func(freq_fit, *(self._params_to_array(self.p1)))

        main_ax, mag_ax, ang_ax = axes

        main_ax.plot(_fit0.real, _fit0.imag, ls=":", color="orange", label="Init. Params.")
        main_ax.plot(_fit1.real, _fit1.imag, ls="-", color="black", label="Fit")

        # Shrink current axis by 20%
        box = main_ax.get_position()
        main_ax.set_position([box.x0 - 0.2 * box.width, box.y0, box.width, box.height])

        # Put a legend to the right of the current axis
        main_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

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
        f0, Qe, Qi = results['f0'][0], results['Qe'][0], results['Qi'][0]
        κe_2pi, κi_2pi = results['kappa_eOver2pi'][0], results['kappa_iOver2pi'][0]

        f0_err = results['f0'][1] - f0
        κe_2pi_err = results['kappa_eOver2pi'][1] - κe_2pi
        κi_2pi_err = results['kappa_iOver2pi'][1] - κi_2pi

        mag_ax.set_title(r"$Q_e = %d$, $Q_i = %d$" % (Qe, Qi))

        n_sigma = self.n_sigma
        main_ax.set_title(r"%.2f%s confidence interval (%d$\sigma$)" % \
                          (100 * erf(n_sigma/np.sqrt(2)), '%', n_sigma))

        # add fit result and confidence interval to the plot
        window = self.__plot_window

        f0_str = r"$f_0 = %.4f_{-%.4f}^{+%.4f}$ GHz" % \
            (f0/1e9, *np.abs(f0_err/1e9))
        κe_2pi_str = r"$\kappa_e/2\pi = %.4f_{-%.4f}^{+%.4f}$ MHz" % \
            (κe_2pi/1e6, *np.abs(κe_2pi_err/1e6))
        κi_2pi_str = r"$\kappa_i/2\pi = %.4f_{-%.4f}^{+%.4f}$ MHz" % \
            (κi_2pi/1e6, *np.abs(κi_2pi_err/1e6))

        textstr = '\n'.join((f0_str, κe_2pi_str, κi_2pi_str))
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        main_ax.text(0.05, 0.95, textstr, transform=main_ax.transAxes,
                     fontsize=10, verticalalignment='top', bbox=props)


class SingleSidedS11Fit(FrequencyDomain):
    """
    Class for imlementing fitting of lineshape in reflection measurement
    """
    def __init__(self, freq, data, df=0, fit_mag_dB=False, plot_mag_dB=False):
        super().__init__(freq, data, df=df, fit_mag_dB=fit_mag_dB, plot_mag_dB=plot_mag_dB)

        self.fit_type = "SingleSidedReflection"

    def fit_func(self, f, f0_MHz, Q, QoverQe, δf, a, ϕ, τ_ns):
        """
        Reflection fit for resonator single-sided coupled to waveguide
        according to relation
        S11 = a * exp(i(ϕ + 2pi * (f - f[0]) * τ)) .* ...
             (1 - 2 * QoverQe * (1 + 2i * δf / f0) / (1 + 2i * Q * (f - f0) / f0))

        Note: If df = 0 (no background) this gives
        S11 = 1 - (2 kappa_e) / (kappa_i + kappa_e + 2i (omega - omega0))
            = (kappa_i - kappa_e + 2i (omega - omega0)) / (kappa_i + kappa_e + 2i (omega - omega0))
        See Aspelmeyer et al, "Cavity Optomechanics", Rev. Mod. Phys. 86, 1391 (2014).
        """
        return (a * np.exp(1j * (ϕ + 2 * np.pi * (f - f[0]) * τ_ns * 1e-9)) *
                (1 - 2 * QoverQe * (1 + 2j * δf / f0_MHz) /
                (1 + 2j * (Q) * (f/1e6 - f0_MHz) / (f0_MHz))))

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
        if (np.max(_angU) - np.min(_angU)) > 2.1 * np.pi:
            # if phase data at start and stop frequencies differ more than 2pi,
            # perform phase subtraction associated with delay
            τ0 = (_angU[-1] - _angU[0]) / ((f[-1] - f[0]))/ (2 * np.pi)

        # Estimate total Q from the FWHM in |mag|^2
        f0, Δf = self._estimate_f0_FWHM()
        QoverQe0 = 0.5 * (1 - np.min(_mag) / a0)
        Q0 = f0 / Δf

        p0_mag, p0_ang = self._prepare_fit_params(f0, Q0, QoverQe0,
                                                  df, a0, ϕ0, τ0)

        self.p0 = p0_mag + p0_ang
        return p0_mag, p0_ang


class WaveguideCoupledS21Fit(FrequencyDomain):
    """
    Class for imlementing fitting of lineshape in reflection measurement
    """
    def __init__(self, freq, data, df=0, fit_mag_dB=False, plot_mag_dB=False):
        super().__init__(freq, data, df=df, fit_mag_dB=fit_mag_dB, plot_mag_dB=plot_mag_dB)

        self.fit_type = "WaveguideCoupledTransmission"

    def fit_func(self, f, f0_MHz, Q, QoverQe, δf, a, ϕ, τ_ns):
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
        return (a * np.exp(1j * (ϕ + 2 * np.pi * (f - f[0]) * τ_ns * 1e-9)) *
                (1 - QoverQe * (1 + 2j * δf / f0_MHz) /
                (1 + 2j * (Q) * (f/1e6 - f0_MHz) / (f0_MHz))))

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
        if (np.max(_angU) - np.min(_angU)) > 2.1 * np.pi:
            # if phase data at start and stop frequencies differ more than 2pi,
            # perform phase subtraction associated with delay
            τ0 = (_angU[-1] - _angU[0]) / ((f[-1] - f[0]))/ (2 * np.pi)

        # Estimate total Q from the FWHM in |mag|^2
        f0, Δf = self._estimate_f0_FWHM()
        QoverQe0 = (1 - np.min(_mag) / a0)
        Q0 = f0 / Δf
        p0_mag, p0_ang = self._prepare_fit_params(f0, Q0, QoverQe0,
                                                  df, a0, ϕ0, τ0)

        self.p0 = p0_mag + p0_ang
        return p0_mag, p0_ang
