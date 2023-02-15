import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.signal import windows
from typing import Optional

from .helper_functions import number_with_si_prefix, si_prefix_to_scaler
from scipy.linalg import svd
import cma

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
        self.projected_data = self._inner_product(self.data - self.v_g, self.n)
        self.population =  self.projected_data / np.abs(self.v_e - self.v_g)

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
        self.amp_norm = np.max(amp) - np.min(amp)
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
    
        dz = (amp - amp0) / self.amp_norm
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
        if self.guess_FFT:
            fft_pop = np.fft.rfft(self.signal, axis=-1)
            fft_pop[:, 0] = 0 ## remove the dc part if any
            fft_freq = np.fft.rfftfreq(len(self.signal[0, :]), d=(self.time[1] - self.time[0]))
            n_amp, n_freq = fft_pop.shape
            freq_max = []
            rel_amp_array = []
            max_fft = np.max(np.max(np.abs(fft_pop)))
            max_freq_ind = n_freq
                
            for j in range(n_amp):
                if np.max(np.abs(fft_pop[j, :])) > 0.15 * max_fft:
                    ind_max_intensity = np.argmax(np.abs(fft_pop[j, :]))
                    freq_max.append(fft_freq[ind_max_intensity])
                    rel_amp_array.append(self.amp[j])
                    if ind_max_intensity <= max_freq_ind:
                        if ind_max_intensity < max_freq_ind:
                            i0_list = [j]
                            max_freq_ind = ind_max_intensity
                        else:
                            i0_list.append(j)
            i0 = int(np.floor(np.mean(i0_list)))

            # i0 = np.argmin(np.argmax(np.abs(fft_pop), axis=-1))
        else:
            i0 = np.argmax(sig_var)
        amp0, sig0 = self.amp[i0], self.signal[i0]
        print(amp0)
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
        c1 = 2 * g0 / ((self.amp[i1] - self.amp[i2]) / self.amp_norm)

        self.p0 = ([g0, gamma0, gamma0, gamma0, gamma0] + [r0, r1, amp0, c1] +
                   [0.0] * (self.amp_polyorder - 1))

    def analyze(self, p0=None, plot=True, guess_FFT=False, **kwargs):
        self.guess_FFT = guess_FFT
        self._set_init_params(p0)

        def lsq_func(params):        
            result_arr = self.fit_func(self.time, self.amp, params)
            
            return (result_arr - self.signal).flatten()

        res = least_squares(
            lsq_func, self.p0,
            bounds=([0.0, 0.0, 0.0, 0.0, 0.0, -np.inf, -np.inf, -np.inf] +
                    [-np.inf] * 1, np.inf), **kwargs)
            # ftol=1e-12, xtol=1e-12)

        self.popt = res.x
        
        # if self.amp_polyorder > 1:
        #     c1 = self.popt[-1]
        #     amp0 = self.popt[7]
        #     sig_var = np.var(self.signal, axis=1)
        #     i0 = np.argmin(np.abs(self.amp - amp0))

        #     sig_var_mid = 0.5 * (np.max(sig_var) + np.min(sig_var))
        #     i1 = np.argmin(np.abs(sig_var[i0:] - sig_var_mid)) + i0
        #     i2 = np.argmin(np.abs(sig_var[:i0] - sig_var_mid))

        #     da_l = (amp0 - self.amp[i2]) / self.amp_norm
        #     da_r = (self.amp[i1] - amp0) / self.amp_norm
        #     c2 = - c1 * (da_l - da_r) / (da_l ** 2 + da_r ** 2) / 10
        #     self.p0 = self.p0 + [c2]

        #     # try:
        #     res = least_squares(
        #         lsq_func, self.p0,
        #         bounds=([0.0, 0.0, 0.0, 0.0, 0.0, -np.inf, -np.inf, -np.inf] +
        #                 [-np.inf] * self.amp_polyorder, np.inf), **kwargs)
            
        #     self.popt = res.x
        #     # except:
        #         # pass
        
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
        
        dz = (self.amp - self.amp0) / self.amp_norm
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
                   cmap=plt.cm.hot)
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
                   shading='auto', cmap=plt.cm.hot)

        plt.axhline(y=self.amp0 / self.amp_scaler, ls='--', color='black')
        plt.xlim(np.min(self.time) / self.time_scaler, np.max(self.time) / self.time_scaler)
        
        _, g_2pi_prefix = number_with_si_prefix(np.max(np.abs(self.g / (2 * np.pi))))
        g_2pi_scaler = si_prefix_to_scaler(g_2pi_prefix)

        plt.tick_params(axis='x', which='both', labelbottom=False)
        plt.yticks(fontsize='x-small')
        plt.ylabel('Amp' +
                   (' (' + self.amp_prefix + ')' if len(self.amp_prefix) > 0 else ''),
                   fontsize='small')

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

class QuantumWalk_1P(TimeDomain):
    def __init__(self, time, pop):
        # initialize parameters
        self.time = time
        self.pop = pop
        self.N_emit = len(pop)
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
        pass

    def _guess_init_params(self):
        """
        Guess initial parameters from data. Will be overwritten in subclass
        """
        pass

    def _set_init_params(self, p0):
        pass
    
    def _save_fit_results(self, popt, pcov):
        pass

    def analyze(self, tij_mat, p0=None, plot=True, plot_cma=False,
                omega_max=(5e6 * 2 * np.pi), sigma=0.7, tolx=0.001):
        """
        Analyze the data with initial parameter `p0`.
        """
        def qw_1Psolver_ED_10ini(t_list, omega_list, tij_mat, N_emit):
            """
            Schrodinger equation solver for quantum walk with a single particle
            N_emit number of emitters.
            """
            H_sub = np.zeros((N_emit, N_emit))
            for j in range(N_emit):
                H_sub[j,j] = omega_list[j]
                for k in range(N_emit):
                    if j != k:
                        H_sub[j,k] = tij_mat[j, k]
            
            v, w = np.linalg.eigh(H_sub)
            pop_list = []
            for jini in range(N_emit):
                psi0_sub = np.zeros((N_emit, 1))
                psi0_sub[jini, :] = 1
                coef = np.matmul(np.transpose(psi0_sub), w)
                tevolve_mat = np.exp(-1j * np.matmul(np.transpose([v]), [t_list]))
                coef_tevolve = np.matmul(np.diag(coef[0]), tevolve_mat)
                evolve_result = np.matmul(w, coef_tevolve)
                pop_list.append((np.abs(evolve_result)) ** 2)
            return pop_list
        
        def cost_fun(times, tij_mat, N_emit, omega_max, pop):
            def simulation_cost(rel_omega_list):
                omega_list = rel_omega_list * omega_max
                result_list = qw_1Psolver_ED_10ini(times, omega_list, tij_mat, N_emit)
                sqr_cost = 0
                for je in range(N_emit):
                    sqr_cost += np.sum((pop[je] - result_list[je])**2)
                # print(f'cost: {sqr_cost}')
                return sqr_cost
            return simulation_cost
        
        if p0 is None:
            p0 = ([0] * self.N_emit)
        self.es = cma.CMAEvolutionStrategy(p0, sigma, {'bounds': [-1, 1], 'tolx': tolx})
        self.es.optimize(cost_fun(self.time, tij_mat, self.N_emit, omega_max, self.pop))
        if plot_cma:
            cma.plot()
        
        self.omega_fit = self.es.result_pretty().xbest * omega_max
        self.result_list = qw_1Psolver_ED_10ini(self.time, self.omega_fit, tij_mat, self.N_emit)
        self.is_analyzed = True
        
        if plot:
            self.plot_result()

    def _plot_base(self):
        pass

    def plot_result(self):
        """
        Will be overwritten in subclass
        """
        if not self.is_analyzed:
            raise ValueError("The data must be analyzed before plotting")
        else:
            q_cent = int((self.N_emit - 1) / 2)
            site = np.array(range(int((self.N_emit - 1) / 2))) + q_cent
            fig, ax = plt.subplots(4, int(self.N_emit / 2), figsize=(20, 7))
            je = 0
            for jx in range(2):
                for jy in range(int(self.N_emit / 2)):
                    c = ax[jx, jy].pcolor(self.time * 1e9, np.arange(self.N_emit + 1) + 0.5,  
                                          self.pop[je], cmap = 'hot')
                    ax[jx, jy].set_xlabel('time (ns)')
                    ax[jx, jy].set_ylabel('Qubit')
                    ax[jx, jy].set_aspect(80)
                    fig.colorbar(c, ax = ax[jx, jy])
                    je += 1
            
            je = 0
            for jx in [2,3]:
                for jy in range(int(self.N_emit / 2)):
                    c = ax[jx, jy].pcolor(self.time * 1e9, np.arange(self.N_emit + 1) + 0.5,  
                                          self.result_list[je], cmap = 'hot')
                    ax[jx, jy].set_xlabel('time (ns)')
                    ax[jx, jy].set_ylabel('Fitting Qubit')
                    ax[jx, jy].set_aspect(80)
                    fig.colorbar(c, ax = ax[jx, jy])
                    je += 1
            
            fig.tight_layout()
            plt.show()
            
            plt.figure()
            plt.plot(range(1, 11, 1), (self.omega_fit - np.mean(self.omega_fit)) / (2 * np.pi * 1e6), 'o', label='cma'); 
            plt.legend()
            plt.xlabel('qubit');
            plt.ylabel('detuning (MHz)')
            plt.pause(0.1)
            plt.draw()
            
        