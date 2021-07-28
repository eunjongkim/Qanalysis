# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 10:36:31 2020

@author: paint
"""

from pandas import DataFrame
from scipy.optimize import least_squares
from scipy.stats import multivariate_normal
from scipy.special import erfc
from sklearn import mixture
import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
import seaborn as sns
from scipy.signal import windows


class SingleShotGaussian:
    """
    Class to implement analysis of single-shot measurement records of
    superconducting qubits.
    
    Parameters
    ----------
    signal: numpy.ndarray
        A 2D numpy array containing readout signals to be analyzed. Each row 
        index i corresponds to the readout signal with qubit prepared in
        state i. The column index j is the index of repetition.

    Attributes
    ----------
    signal : numpy.ndarray
        A 2D numpy array containing readout signals to be analyzed.
    
    num_of_states : int
        Number of states in the readout analysis, which determines the number
        of Gaussians assumed in the fitting. This is extracted from the number
        of rows of the input argument `signal`.

    num_of_points : int
        Number of points in the analysis. This is obtained from the number of
        columns of the input argument `signal`.

    means : numpy.ndarray
        A `num_of_states`-sized 1D array containing the mean of each Gaussian
        distribution (a complex number) corresponding to the states extracted
        from the fitting.

    variances : numpy.ndarray
        A 1D array of size `num_of_states` containing the variance of each
        Gaussian distribution (a positive number) corresponding to the states
        extracted from the fitting.

    signal_to_noise_ratio : dict
        A dictionary containing the mapping of a pair (i, j) of states to 
        signal-to-noise ratio (amplitude ratio) of discriminating the states
        i and j.

    weights: numpy.ndarray
        A 2D array of shape `(num_of_states, num_of_states)` containing the
        weight of Gaussian modes extracted from the fitting.

    state_prediction : numpy.ndarray
        A 2D array of shape `(num_of_states, num_of_points)` containing the 
        state assignment result of the signal. Each readout signal assigned to
        the state that maximizes the log probability of fitted Gaussian
        distributions.

    confusion : numpy.ndarray
        A 2D array of shape `(num_of_states, num_of_states)` containing the
        confusion matrix, which shows the probability P(j|i) that the qubit
        is predicted to be in state j (column index) under preparation of
        state i (row index).

    fidelity : float
        Readout fidelity. This is defined as the average of diagonal elements
        of the confusion matrix.

    Methods
    -------
    analyze(binsI=20, binsQ=20, cluster_method='gmm', plot=True)
        Perform analysis of the readout data.
    
    plot_result()
        Plot the result of the single-shot readout analysis.
    
    """

    _bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)

    def __init__(self, signal):
        # An array of complex readout signals when the qubit was initialized to state 0, state 1, ...
        self.signal = signal
        self.num_of_states, self.num_of_points = signal.shape
        self.is_analyzed = False
        self.means = None
        self.variances = None
        self.weights = None
        self.signal_to_noise_ratio = None

        self.hists = None
        self.hist_ranges = None
        self.hist_fits = None

        self.state_prediction = None
        self.confusion = None
        self.fidelity = None

    def analyze(self, binsI=50, binsQ=50, cluster_method='gmm', plot=True):
        """
        Analyze the single-shot readout data by fitting the full data to
        a sum of Gaussian distributions.

        Parameters
        ----------
        binsI : int, optional
            The number of bins along the I axis (real) to evaluate histogram.
            The default is 50.
        binsQ : int, optional
            The number of bins along the Q axis (imag) to evaluate histogram.
            The default is 50.
        cluster_method : str, optional
            Method to be used for initial clustering of the data.
            The default is 'gmm'.
        plot : bool, optional
            If True, the result of the analysis is plotted automatically.
            If False, the analysis result can be popped up by calling the
            `plot_result` method. The default is True.

        Returns
        -------
        None.

        """
        if cluster_method == 'gmm':
            self._gaussian_mixture_clustering()

            # fitting with multi-mode Gaussian
        x0, bounds = self._prepare_x0_bounds()
        fit = least_squares(self._gaussian_mixture_residual_function, x0,
                            bounds=bounds,
                            args=(binsI, binsQ))
        self.means, self.variances, self.weights = self._x0_to_meas_vars_weights(fit.x)

        # fidelity estimation
        logprob = np.zeros((self.num_of_states,
                            self.num_of_points,
                            self.num_of_states))
        for i in range(self.num_of_states):
            logprob[:, :, i] = \
                (- (np.abs(self.signal - self.means[i]) ** 2 /
                    (2 * self.variances[i])) - np.log(self.variances[i]))
        # choose the state maximizing the log probablility
        self.state_prediction = np.argmax(logprob, axis=-1)

        # confusion matrix & fidelity
        confusion = (self._count_occurences(self.state_prediction) /
                     self.num_of_points)
        self.confusion = confusion
        self.fidelity = np.sum(np.diagonal(confusion)) / self.num_of_states

        # SNR
        comb_list = list(combinations(range(self.num_of_states), 2))
        snr_list = []
        for c in comb_list:
            s1, s2 = c
            signal = np.abs(self.means[s1] - self.means[s2])
            noise = np.sqrt(0.5 * (self.variances[s1] + self.variances[s2]))
            snr = signal / noise
            snr_list.append(snr)
        self.signal_to_noise_ratio = dict(zip(comb_list, snr_list))

        self.is_analyzed = True
        if plot:
            self.plot_result()

    def _prepare_x0_bounds(self):
        means_flat = np.dstack((self.means.real, self.means.imag))
        means_vars = np.append(means_flat, self.variances)
        x0 = np.append(means_vars, self.weights[1:, :].flatten())

        lb = []
        ub = []
        sig_flat = self.signal.flatten()
        minI, minQ = np.min(sig_flat.real), np.min(sig_flat.imag)
        maxI, maxQ = np.max(sig_flat.real), np.max(sig_flat.imag)
        for i in range(self.num_of_states):
            lb.append(minI)
            lb.append(minQ)
            ub.append(maxI)
            ub.append(maxQ)
        for i in range(self.num_of_states):
            lb.append(0)
            ub.append(np.max([maxI - minI, maxQ - minQ]))
        for i in range(self.num_of_states * (self.num_of_states - 1)):
            lb.append(0)
            ub.append(1)
        bounds = (lb, ub)
        return x0, bounds

    def _x0_to_meas_vars_weights(self, x0):
        means_ = x0[:(2 * self.num_of_states)].reshape(self.num_of_states, 2)
        means = means_[:, 0] + 1j * means_[:, 1]
        variances = x0[(2 * self.num_of_states):(3 * self.num_of_states)]

        weights = np.zeros((self.num_of_states, self.num_of_states))
        weights[:, 1:] = x0[(3 * self.num_of_states):].reshape(self.num_of_states,
                                                               self.num_of_states - 1)
        weights[:, 0] = 1 - np.sum(weights[:, 1:], axis=1)
        return means, variances, weights

    def _gaussian_mixture_residual_function(self, x0, binsI, binsQ):
        means, variances, weights = self._x0_to_meas_vars_weights(x0)

        sig_flat = self.signal.flatten()
        hist_range = [[np.min(sig_flat.real), np.max(sig_flat.real)],
                      [np.min(sig_flat.imag), np.max(sig_flat.imag)]]

        hists = []
        I_edges, Q_edges = None, None
        for i in range(self.num_of_states):
            sig = self.signal[i, :]
            H, I_edges, Q_edges = np.histogram2d(sig.real, sig.imag,
                                                 bins=[binsI, binsQ],
                                                 range=hist_range, normed=True)
            hists.append(H)
        self.hists = np.array(hists)

        I_range = (I_edges[1:] + I_edges[:-1]) / 2
        Q_range = (Q_edges[1:] + Q_edges[:-1]) / 2
        self.hist_ranges = I_range, Q_range

        # create a binsI X binsQ grid of points in the (I, Q) plane
        I_, Q_ = np.meshgrid(I_range, Q_range)
        pos = np.dstack((I_.T, Q_.T))

        # 2D normal distribution
        norm_dists = []
        for i in range(self.num_of_states):
            mu = [means[i].real, means[i].imag]
            var = variances[i]
            rv = multivariate_normal(mean=mu, cov=np.diag([var, var]))
            pdf_ = rv.pdf(pos)
            norm_dists.append(pdf_)
        norm_dists = np.array(norm_dists)
        self.norm_dists = norm_dists

        self.hist_fits = np.tensordot(weights, norm_dists, axes=([1], [0]))

        return (self.hists - self.hist_fits).flatten()

    def _count_occurences(self, pred):

        count = np.zeros((self.num_of_states,
                          self.num_of_states))
        for i in range(self.num_of_states):
            for j in range(self.num_of_states):
                count[i, j] = np.sum(pred[i, :] == j)
        return count

    def _gaussian_mixture_clustering(self):
        # perform clustering of the full data set with Gaussian mixture model
        sig_flat = self.signal.flatten()
        data = {'I': sig_flat.real, 'Q': sig_flat.imag}
        x = DataFrame(data, columns=['I', 'Q'])

        # train the dataset with Gaussian Mixture model
        gmm = mixture.GaussianMixture(n_components=self.num_of_states,
                                      covariance_type='spherical', tol=1e-12,
                                      reg_covar=1e-12).fit(x)
        # predict the states
        pr_state_ = gmm.predict(x).reshape(self.num_of_states, self.num_of_points)
        bincounts = self._count_occurences(pr_state_)
        # create mapping of gmm components to state labels
        mapping = np.argsort([np.argmax(bincounts[:, i]) for i in range(self.num_of_states)])

        # reorder the components of the fit to state indices
        means_ = gmm.means_
        self.means = np.array([means_[i, 0] + 1j * means_[i, 1] for i in range(self.num_of_states)])[mapping]
        self.variances = np.array([gmm.covariances_[mapping][0] for i in range(self.num_of_states)])
        self.state_prediction = pr_state_[mapping, :]

        # weights of each state preparation fitted to Gaussian components
        self.weights = (self._count_occurences(self.state_prediction) /
                        self.num_of_points)

    def _plot_iq_blobs(self, blob_ax):
        """
        Plot IQ blobs in the complex plane with SNR info annotated.
        """
        state_markers = ['o', 'v', 's', 'p', '*', 'h', '8', 'D']
        data_ms = np.sqrt(10000 / self.num_of_points)

        for i in range(self.num_of_states):
            blob_ax.plot(self.signal[i, :].real, self.signal[i, :].imag,
                         '.', label=r'$|%d\rangle$ Prep.' % i, alpha=.3,
                         color='C%d' % i, ms=data_ms)
        for i in range(self.num_of_states):
            blob_ax.plot(self.means[i].real, self.means[i].imag,
                         marker=state_markers[i], ms=5, color='black')
        blob_ax.set_xlabel("I")
        blob_ax.set_ylabel("Q")
        blob_ax.legend(fontsize='xx-small', loc=2)

        snr_str = []
        for k in self.signal_to_noise_ratio.keys():
            s1, s2 = k
            snr = self.signal_to_noise_ratio[k]
            snr_str.append(r"$|%d\rangle$-$|%d\rangle$ SNR: %.3f" % (s1, s2, snr))

        blob_ax.text(1, 1, "\n".join(snr_str), size=6,
                     transform=blob_ax.transAxes, ha="right", va="top",
                     bbox=self._bbox_props)

        for i in range(self.num_of_states):
            mu = self.means[i]
            sigma = np.sqrt(self.variances[i])
            circ3sigma = plt.Circle((mu.real, mu.imag), 3 * sigma, lw=1,
                                    color='C%d' % i, fill=False)
            blob_ax.add_artist(circ3sigma)
        blob_ax.axis('equal')

    def _plot_confusion_matrix(self, conf_mat_ax):
        """
        Plot confusion matrix & readout fidelity
        """
        labels = [r'$|%d\rangle$' % i for i in range(self.num_of_states)]
        sns.heatmap(self.confusion, annot=True, ax=conf_mat_ax,
                    fmt='g', cmap='Blues')
        # labels, title and ticks
        conf_mat_ax.set_ylabel('Preparation')
        conf_mat_ax.set_xlabel('Prediction')
        conf_mat_ax.set_title(r'Readout fidelity $\mathcal{F}=%.3f$' % self.fidelity)
        conf_mat_ax.xaxis.set_ticklabels(labels)
        conf_mat_ax.yaxis.set_ticklabels(labels)

    def plot_result(self):
        """
        Plot th result of the readout analysis.

        Returns
        -------
        fig : matplotlib.figure.Figure
            A figure with the readout analysis result.
        """
        fig = plt.figure(constrained_layout=True)
        # gridspec
        gs = fig.add_gridspec(2, 1)
        # subgridspec
        sgs0 = gs[0].subgridspec(2, self.num_of_states)
        sgs1 = gs[1].subgridspec(1, 2)

        data_axes = [fig.add_subplot(sgs0[0, i]) for i in range(self.num_of_states)]
        fit_axes = [fig.add_subplot(sgs0[1, i]) for i in range(self.num_of_states)]
        blob_ax = fig.add_subplot(sgs1[0])
        conf_mat_ax = fig.add_subplot(sgs1[1])

        # plot data & fit of each state preparation result
        data_axes[0].set_ylabel("Data")
        fit_axes[0].set_ylabel("Fit")
        for i in range(self.num_of_states):
            data_axes[i].contourf(*self.hist_ranges, self.hists[i].T, levels=100)
            fit_axes[i].contourf(*self.hist_ranges, self.hist_fits[i].T, levels=100)
            weight_str = [r"$w_{|%d\rangle} = %.3f$" % (k, self.weights[i, k]) for k in range(self.num_of_states)]

            data_axes[i].set_title(r"$|%d\rangle$ Prep." % i)

            fit_axes[i].text(1, 1, "\n".join(weight_str), size=6,
                             transform=fit_axes[i].transAxes,
                             ha="right", va="top",
                             bbox=self._bbox_props)
            data_axes[i].axis('equal')
            fit_axes[i].axis('equal')

        # plot IQ blobs with SNR
        self._plot_iq_blobs(blob_ax)

        # plot confusion matrix & readout fidelity
        self._plot_confusion_matrix(conf_mat_ax)

        return fig


class SingleShotGaussianTwoStates(SingleShotGaussian):
    def __init__(self, signal):
        super().__init__(signal)
        if self.num_of_states != 2:
            raise ValueError("The number of states (number of rows of input argument) to be analyzed is 2.")

    def analyze(self, binsI=50, binsQ=50, cluster_method='gmm', plot=True):
        # run analysis of the superclass without plotting
        super().analyze(binsI=binsI, binsQ=binsQ,
                        cluster_method=cluster_method, plot=False)

        # perform projection to 1D line connecting the means of two fitted Gaussians
        self._project_to_1D()

        if plot:
            self.plot_result()

    def _inner_product(self, z1, z2):
        """
        Element-wise inner product between complex vectors or between a
        complex vector and a complex number.
        """
        return z1.real * z2.real + z1.imag * z2.imag

    def _calculate_desc_bd(self):
        """
        Calculate the decisiion booundary.

        Returns
        -------
        float
            Location of the decision boundary maximizing the likelihood.

        """
        mu0, mu1 = self.means1D
        var0, var1 = self.variances1D

        a = 0.5 * (1 / var0 - 1 / var1)
        b = - (mu0 / var0 - mu1 / var1)
        c = 0.5 * (mu0 ** 2 / var0 - mu1 ** 2 / var1) - 0.5 * np.log(var1 / var0)

        if a == 0:
            return -c / b
        else:
            return (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)

    def _project_to_1D(self):

        mu0, mu1 = self.means
        self.origin = 0.5 * (mu0 + mu1)  # origin of the axis of projection

        # unit vector along the line connecting the mean of Gaussians
        self.n = (mu1 - mu0) / np.abs(mu1 - mu0)
        sigma = np.sqrt(np.mean(self.variances))

        # projection of complex data to a line origin + x * n
        self.signal1D = np.vstack([self._inner_product(self.signal[i] - self.origin,
                                                       self.n) / sigma for i in range(self.num_of_states)])
        self.means1D = self._inner_product(self.means - self.origin, self.n) / sigma
        self.variances1D = self.variances / sigma ** 2

        self.decision_boundary1D = self._calculate_desc_bd()
        self.decision_boundary = (self.decision_boundary1D * sigma * self.n +
                                  self.origin)

        _hist_range = np.min(self.signal1D), np.max(self.signal1D)
        hists1D = []
        s_edges = None
        for i in range(self.num_of_states):
            H, s_edges = np.histogram(self.signal1D[i, :], bins=100,
                                      range=_hist_range, density=True)
            hists1D.append(H)
        self.hists1D = np.array(hists1D)
        self.hist_range1D = (s_edges[1:] + s_edges[:-1]) / 2

        # 1D normal distribution
        norm_dists1D = []
        for i in range(self.num_of_states):
            rv = multivariate_normal(mean=self.means1D[i],
                                     cov=self.variances1D[i])
            pdf_ = rv.pdf(self.hist_range1D)
            norm_dists1D.append(pdf_)

        self.norm_dists1D = np.array(norm_dists1D)
        self.hist_fits1D = np.tensordot(self.weights, self.norm_dists1D,
                                        axes=([1], [0]))
        # overlap error is calculated from the distance of the mean from the
        # decision boundary normalized to sqrt(2) * sigma
        t = (np.abs(self.means1D - self.decision_boundary1D) /
             np.sqrt(2 * self.variances1D))
        self.overlap_error = erfc(t) / 2
        self.residual_error = np.array([
            self.confusion[0, 1] - self.overlap_error[0],
            self.confusion[1, 0] - self.overlap_error[1]])

    def _plot_iq_blobs(self, blob_ax):
        super()._plot_iq_blobs(blob_ax)

        I_range, Q_range = self.hist_ranges
        Imin, Imax = np.min(I_range), np.max(I_range)
        Qmin, Qmax = np.min(Q_range), np.max(Q_range)

        # draw a line to which projection of signal was performed

        # slope = self.n.imag / self.n.real
        # x0, y0 = self.origin.real, self.origin.imag
        # project_line = slope * (I_range - x0) + y0
        # mask0 = (project_line < Qmax) * (project_line > Qmin)
        # I_proj, Q_proj =  I_range[mask0], project_line[mask0]
        # if len(I_proj) < 2:
        #     # if projection line is nearly vertical
        #     Q_proj = np.linspace(Qmin, Qmax, 10)
        #     I_proj = (Q_proj - y0) / slope + x0
        # blob_ax.plot(I_proj, Q_proj, color='black', lw=1)
        blob_ax.axline((self.means[0].real, self.means[0].imag),
                       (self.means[1].real, self.means[1].imag),
                       color='black', lw=1)

        # draw a decision boundary to for state assignment
        t = - self.n.imag + 1j * self.n.real
        x1, y1 = self.decision_boundary.real, self.decision_boundary.imag
        # x1, y1 = self.decision_boundary.real, self.decision_boundary.imag
        # decision_line = slope1 * (I_range - x1) + y1
        # mask1 = (decision_line < Qmax) * (decision_line > Qmin)
        # I_dec, Q_dec =  I_range[mask1], decision_line[mask1]
        # if len(I_dec) < 2:
        #     # if decision boundary is nearly vertical
        #     Q_dec = np.linspace(Qmin, Qmax, 10)
        #     I_dec = (Q_dec - y1) / slope1 + x1
        # blob_ax.plot(I_dec, Q_dec, color='black', ls='--', lw=1)
        d = np.abs(self.means[1] - self.means[0])
        p0 = self.decision_boundary - t * 0.5 * d
        p1 = self.decision_boundary + t * 0.5 * d
        blob_ax.axline((p0.real, p0.imag), (p1.real, p1.imag),
                       color='black', ls='--', lw=1)

    def plot_result(self):
        fig = plt.figure(constrained_layout=True)
        # gridspec
        gs = fig.add_gridspec(2, 1)
        data_fit_ax = fig.add_subplot(gs[0])

        sgs1 = gs[1].subgridspec(1, 2)
        blob_ax = fig.add_subplot(sgs1[0])
        conf_mat_ax = fig.add_subplot(sgs1[1])

        width = self.hist_range1D[1] - self.hist_range1D[0]
        for i in range(self.num_of_states):
            data_fit_ax.bar(self.hist_range1D, self.hists1D[i], width=width,
                            color="C%d" % i, alpha=0.5,
                            label=r"$|%d\rangle$ Data" % i)
        for i in range(self.num_of_states):
            data_fit_ax.plot(self.hist_range1D, self.hist_fits1D[i],
                             color="C%d" % i, lw=2,
                             label=r"$|%d\rangle$ Fit" % i)
            data_fit_ax.plot(self.hist_range1D, self.norm_dists1D[i],
                             color='black', ls=':', lw=1)
        data_fit_ax.set_yscale("log")
        data_fit_ax.set_xlabel(r"$\Delta s/\sigma$")
        data_fit_ax.set_ylabel(r"Prob. Density $P(\Delta s/\sigma)$")
        data_fit_ax.set_ylim([1e-4, 1])
        data_fit_ax.axvline(self.decision_boundary1D, ls='--', color='black',
                            label="Decision Bd.")
        data_fit_ax.legend(fontsize='x-small', loc=2)

        error_str = []
        for i in range(self.num_of_states):
            error_str.append(r"Overlap err. $|%d\rangle$: %.4f" % (i, self.overlap_error[i]))
        for i in range(self.num_of_states):
            error_str.append(r"Residual err. $|%d\rangle$: %.4f" % (i, self.residual_error[i]))

        data_fit_ax.text(1, 1, "\n".join(error_str), size=6,
                         transform=data_fit_ax.transAxes, ha="right", va="top",
                         bbox=self._bbox_props)

        # plot IQ blobs with SNR
        self._plot_iq_blobs(blob_ax)

        # plot confusion matrix & readout fidelity
        self._plot_confusion_matrix(conf_mat_ax)

        return fig

class ReadoutTrace:
    def __init__(self, adc, frequency: float, adc_sample_rate: float=1e9,
                 downsample_factor: int=4, timediff: int=0,
                 prediction=None, timestamp=None, demod_factor: float=2 ** -12):
        self.adc_trace = adc
        self.frequency = frequency
        self.adc_sample_rate = adc_sample_rate
        self.downsample_factor = downsample_factor
        
        if len(adc.shape) == 2:
            self.num_of_states, self.readout_length = adc.shape
        if len(adc.shape) == 3:
            self.num_of_states, self.num_of_points, self.readout_length = adc.shape
        self.times = np.arange(self.readout_length) / self.adc_sample_rate
        self.demod_factor = demod_factor  # constant factor multiplied during demodulation
        # prepare timestamps of all adc datapoints
        self.timestamp = timestamp
        self.timediff = timediff
        self._prepare_timestamp()
        self.prediction = prediction
        if prediction is None:
            self.prediction = np.outer(np.arange(self.num_of_states),
                                       np.ones(self.num_of_points))

    def _prepare_timestamp(self):
        if self.timestamp is None:
            # if timestamp is not provided, assume that all adc traces are
            # starting at zero initial phase and create an array of zero
            self.timestamp = np.tensordot(
                np.ones((self.num_of_states, self.num_of_points)),
                np.arange(self.readout_length), axes=0)

        else:
            if len(self.timestamp.shape) == 2:
                self.timestamp = \
                    (np.tensordot(
                        self.timestamp, np.ones(self.readout_length), axes=0) +
                     np.tensordot(
                         np.ones((self.num_of_states, self.num_of_points)),
                         np.arange(self.readout_length), axes=0))

    def _downconvert_adc_trace(self, filter_bandwidth):
        self.downconverted_adc_trace = \
            self.adc_trace * np.exp(-1j * 2 * np.pi * self.frequency *
                                    (self.timestamp - self.timediff) /
                                    self.adc_sample_rate)

        if filter_bandwidth is not None:
            self.bandwidth = filter_bandwidth
            # number of samples corresponding to the bandwidth of readout pulse
            time_window = int(1 / np.abs(self.bandwidth) * self.adc_sample_rate)
            # Hann window with (frequency / 2) cutoff
            hann_ = windows.hann(time_window * 2, sym=True)
            hann_ = hann_ / np.sum(hann_)

            for k in range(self.num_of_states):
                for l in range(self.num_of_points):
                    self.downconverted_adc_trace[k, l, :] = \
                        np.convolve(self.downconverted_adc_trace[k, l, :],
                                    hann_, 'same')

    def _average_downconverted_adc_trace(self):
        self.avg_downconverted_adc_trace = np.zeros((self.num_of_states,
                                                     self.readout_length),
                                                    dtype=complex)
        for i in range(self.num_of_states):
            mask = (self.prediction[i, :] == i)
            self.avg_downconverted_adc_trace[i, :] = \
                np.mean(self.downconverted_adc_trace[i, mask, :], axis=-2)

    def _downsample(self):
        downsampled_traces = []

        samples = self.downsample_factor
        traces = self.avg_downconverted_adc_trace
        # downsample from adc sample rate to integration weight sample rate
        for i in range(self.num_of_states):
            downsampled_traces.append(np.mean(np.reshape(traces[i, :],
                                                         (-1, samples)),
                                              axis=1))
        self.downsampled_avg_traces = np.array(downsampled_traces)

    def _find_weight_and_bias(self):
        traces = self.downsampled_avg_traces
        norm = np.max(np.abs(traces))
        self.weights = traces / norm
        self.bias = (np.linalg.norm(self.weights, axis=1, ord=2) ** 2 / 2 *
                     norm * self.downsample_factor * self.demod_factor)

        if self.num_of_states == 2:
            self.optimal_integration_weight = self.weights[1, :] - self.weights[0, :]
            self.expected_demodulated_points = \
                (np.conj(self.optimal_integration_weight) @ traces.T *
                 self.downsample_factor * self.demod_factor)
            self.optimal_bias = self.bias[1] - self.bias[0]

        if self.num_of_states > 2:
            ## Process optimal integration weight based on multiclass LDA for 
            ## dimensionality reduction.
            # Between-class scatter matrix Sb
            mean_trace = np.mean(traces, axis=0)
            Sb = np.transpose(traces - mean_trace) @ np.conj(traces - mean_trace)
            # Sb = np.transpose(traces) @ np.conj(traces)
            # Assume within-class covariance matrix are spherical (diagonal)
            # and identical between classes

            eigval, eigv = np.linalg.eig(Sb)
            # choose eigenvector with largest eigenvalue
            sortperm = np.argsort(np.abs(eigval))
            v = eigv[:, sortperm[-1]]
            norm = np.max(np.abs(v))

            self.optimal_integration_weight = v / norm
            # IQ-point corresponding to readout signal of each state
            # demodulated with the optimal integration weight
            self.expected_demodulated_points = \
                (np.conj(self.optimal_integration_weight) @ traces.T *
                 self.downsample_factor * self.demod_factor)

            self.optimal_bias = (np.abs(self.expected_demodulated_points) ** 2) / 2

    def analyze(self, plot=True, filter_bandwidth=None):

        self._downconvert_adc_trace(filter_bandwidth=filter_bandwidth)
        self._average_downconverted_adc_trace()
        self._downsample()
        self._find_weight_and_bias()

        if plot:
            self.plot_result()

    def plot_result(self):

        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(1, 2)

        sgs = gs[0].subgridspec(2, 1)
        time_axes = [fig.add_subplot(sgs[k]) for k in range(2)]
        iq_ax = fig.add_subplot(gs[1])
        times = self.times

        state_markers = ['o', 'v', 's', 'p', '*', 'h', '8', 'D']
        for i in range(self.num_of_states):
            time_axes[0].plot(times / 1e-9,
                              self.avg_downconverted_adc_trace[i, :].real,
                              color='C%d' % i, ls='-', lw=2)
            # time_axes[0].plot(times / 1e-9,
            #                   self.downconverted_adc_trace[i, 0, :].real,
            #                   color='C%d' % i, ls='-', alpha=0.5, lw=1)

            time_axes[1].plot(times / 1e-9,
                              self.avg_downconverted_adc_trace[i, :].imag,
                              color='C%d' % i, ls='-', lw=2)
            # time_axes[1].plot(times / 1e-9,
            #                   self.downconverted_adc_trace[i, 0, :].imag,
            #                   color='C%d' % i, ls='-', alpha=0.5, lw=1)
            iq_ax.plot(self.avg_downconverted_adc_trace[i, :].real,
                       self.avg_downconverted_adc_trace[i, :].imag,
                       '.-', ms=4, color='C%d' % i,
                       label=r"$|%d\rangle$ Avg. trace" % i)

        demod = np.mean(self.avg_downconverted_adc_trace, axis=-1)
        # weighted_demod = \
        #     np.array([np.average(self.avg_downconverted_adc_trace[i], axis=-1,
        #                          weights=self.weights[i]) for i in range(self.num_of_states)])
        for i in range(self.num_of_states):
            iq_ax.plot(demod[i].real, demod[i].imag,
                       marker=state_markers[i], ms=7, lw=0, color='black',
                       label=r"$|%d\rangle$ Demod." % i)
            # iq_ax.plot(weighted_demod[i].real, weighted_demod[i].imag, 
            #            marker=state_markers[i], ms=7, lw=0, color='blue',
            #            label=r"$|%d\rangle$ Weighted Demod." % i)

        time_axes[1].set_xlabel("Time (ns)")
        time_axes[0].set_ylabel("I (V)")
        time_axes[1].set_ylabel("Q (V)")

        iq_ax.set_xlabel("I (V)")
        iq_ax.set_ylabel("Q (V)")
        iq_ax.axis('equal')
        iq_ax.legend(loc=2, fontsize='x-small')
