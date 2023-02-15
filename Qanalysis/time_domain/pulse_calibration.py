
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