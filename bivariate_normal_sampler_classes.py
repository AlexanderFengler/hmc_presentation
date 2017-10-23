import numdifftools as nd
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
import statsmodels.api as sm

class bivariate_normal_metrop(object):
    def __init__(self,
                 covariance_matrix_target = (np.matrix([[1, 0], [0, 1]])),
                 init = [0, 0],
                 sd_sampler = 1,
                 n_samples = 200):
        self.cov_mat_target = covariance_matrix_target
        self.sd_sampler = sd_sampler
        self.n_samples = n_samples
        self.init = init
        self.sample = 'No sample yet, run the sampler first'
        self.acceptance_rate = 'No sample yet, run the sampler first'
        self.type_of_sampler = 'metropolis'

    def run_sampler(self, feedback_after_n_samples = 100000):
        def log_density_ratio_bivariate_normal(proposal, current, cov_mat_inv):
            num = - (1/2) * np.dot(np.dot(proposal.T, cov_mat_inv), proposal)
            denom =  - (1/2) * np.dot(np.dot(current.T, cov_mat_inv), current)
            return num - denom

        cov_mat_inv = np.linalg.inv(self.cov_mat_target)
        len_pos = len(self.init)
        chain = np.zeros((len_pos + 1, self.n_samples))
        chain[:len_pos, 0] = list(self.init)

        for i in range(self.n_samples):
            proposal = np.matrix(chain[:len_pos, i - 1] + np.random.normal(0, self.sd_sampler, 2))
            if np.log(np.random.uniform()) < log_density_ratio_bivariate_normal(proposal.T,
                                                                    np.matrix(chain[:len_pos, i - 1]).T,
                                                                    cov_mat_inv):
                chain[:len_pos, i] = proposal
                chain[len_pos, i] = 1
            else:
                chain[:len_pos, i] = chain[:len_pos, i - 1]
                chain[len_pos, i] = 0
            if i % feedback_after_n_samples == 0 and i > 0:
                print(str(feedback_after_n_samples) + ' samples completed')
        # Add sample to our class instance
        self.sample = chain.T
        self.acceptance_rate = sum(chain[2,:]) / len(chain[2,:])
        self.effective_n = self.get_effective_sample_size()

    def get_effective_sample_size(self):
        autocorrelation_times = list()
        for i in range(len(self.sample[0,:]) - 2):
            sample_acf = sm.tsa.stattools.acf(self.sample[:, i])
            tmp_autocorrelation_time = 1
            for k in sample_acf[1:]: # starting from one since lag-zero doesn't make sense
                if abs(k) > 0.0999:
                    tmp_autocorrelation_time += 2 * abs(k)
                else:
                    break
            autocorrelation_times.append(tmp_autocorrelation_time)
        effective_n_list = self.n_samples / np.array(autocorrelation_times)
        return min(effective_n_list)

    def plot_ellipse(self, ax, color = 'red'):
        circle = self.make_circle()
        cov_eigenvalues, cov_eigenvectors = np.linalg.eig(self.cov_mat_target)
        # applying the transform to get the correct ellipse for our given covariance matrix
        ellipse = np.dot(np.dot(cov_eigenvectors, np.diag(cov_eigenvalues ** (1/2))), circle.T).T
        # initialize plot and add ellipse
        return ax.plot(ellipse[:, 0], ellipse[:, 1], color = color)

    def make_circle(self):
        def unit_circle_values(t, r):
            return [r * math.cos(t), r*math.sin(t)]
        circle = np.zeros((200,2))
        for i in range(200):
            r = 2 # radius 2 implies that we draw the circle two standard deviations out from the center of a standard bivariate normal
            t = 2 * math.pi * (i / 200)
            circle[i, :] = unit_circle_values(t, r)
        return circle

    def plot_sample(self, annotate = False, show_data = True, save = False, alpha = 0.2):
        # initialize figure
        fig, ax = plt.subplots()
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])

        # plot ellipse into figure
        self.plot_ellipse(ax)
        # add sample to figure
        if show_data:
            ax.plot(self.sample[:, 0], self.sample[:, 1], 'b.', alpha = alpha)

        if annotate:
            # annotate with parameters
            props = dict(boxstyle='square', facecolor='white', alpha=0.5)

            # defined text for annotation
            my_text = 'var_ratio: ' + format(1 / self.cov_mat_target[0, 0], '.1f') + '\n' + 'n: ' + str(self.n_samples) + '\n' + 'sd: ' + str(self.sd_sampler) + '\n' + 'acc: ' + format(self.acceptance_rate, '.2f')
            my_text += '\n' + 'n_eff: ' + str(int(self.effective_n))
            # place a text box in upper left in axes coords
            ax.text(0.05, 0.95, my_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
            ax.set_title("Metropolis Sample")
        # save plot
        if save:
            plt.savefig('metrop_sample_biv_gauss_sample_vratio_' + format(1 / self.cov_mat_target[0,0], '.1f') + '.png')
        # show plot
        plt.show()


    def plot_autocorrelation(self, n_param, save = False):
        # get autocorrelation by parameter
        sample_acf = sm.tsa.stattools.acf(self.sample[:, n_param])
        lags = np.arange(1, len(sample_acf), 1)
        fig, ax = plt.subplots()

        # annotate with parameters
        props = dict(boxstyle='square', facecolor='white', alpha=0.5)

        # defined text for annotation
        my_text = 'vratio: ' + format(1 / self.cov_mat_target[0, 0], '.1f') + '\n' + 'n: ' + str(self.n_samples) + '\n' + 'sd: ' + str(self.sd_sampler)
        my_text += '\n' + 'acc: ' + format(self.acceptance_rate, '.2f')
        my_text += '\n' + 'n_eff: ' + str(int(self.effective_n))
        # place a text box in upper left in axes coords
        ax.text(0.05, 0.45, my_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        ax.set_title("Metropolis Autocorrelation")
        ax.bar(lags, sample_acf[1:], color = 'b')
        ax.set_ylim([-1, 1])

        if save:
            plt.savefig('metrop_sample_biv_gauss_autocorr_1overcorr_' + format(1 / self.cov_mat_target[0,1], '.1f') + '_vratio_' + format(1 / self.cov_mat_target[0,0], '.1f') + '_sd_sampler_' + str(self.sd_sampler) + '.png')
        plt.show()

class bivariate_normal_gibbs(object):
    def __init__(self,
                 covariance_matrix_target = (np.matrix([[1, 0], [0, 1]])),
                 init = [0, 0],
                 n_samples = 200):

        self.cov_mat_target = covariance_matrix_target
        self.n_samples = n_samples
        self.init = init
        self.sample = 'No sample yet, run the sampler first'
        self.effective_n = 'No sample yet, run the sampler first'
        self.acceptance_rate = 1
        self.type_of_sampler = 'gibbs'


    def run_sampler(self, feedback_after_n_samples = 100000):
        cov_xy = self.cov_mat_target[0,1]
        var_y = self.cov_mat_target[0,0]
        var_x = self.cov_mat_target[1,1]
        chain = np.zeros((2, self.n_samples))
        chain[:,0] = list(self.init)

        for i in range(1, self.n_samples, 1):
            chain[0, i] = cov_xy * (1 / var_x) * chain[1, i - 1] + np.sqrt(var_y - ((cov_xy ** 2) * (1 / var_x))) * np.random.normal()
            chain[1, i] = cov_xy * (1 / var_y) * chain[0, i] + np.sqrt(var_x - ((cov_xy ** 2) * (1 / var_y))) * np.random.normal()
            if i % feedback_after_n_samples == 0 and i > 0:
                print(str(feedback_after_n_samples) + ' samples completed')
        self.sample = chain.T
        self.effective_n = self.get_effective_sample_size()

    def get_effective_sample_size(self):
        autocorrelation_times = list()
        for i in range(len(self.sample[0,:]) - 1):
            sample_acf = sm.tsa.stattools.acf(self.sample[:, i])
            tmp_autocorrelation_time = 1
            for k in sample_acf[1:]: # starting from one since lag-zero doesn't make sense
                if abs(k) > 0.0999:
                    tmp_autocorrelation_time += 2 * abs(k)
                else:
                    break
            autocorrelation_times.append(tmp_autocorrelation_time)
        effective_n_list = self.n_samples / np.array(autocorrelation_times)
        return min(effective_n_list)

    def plot_ellipse(self, ax):
        circle = self.make_circle()
        cov_eigenvalues, cov_eigenvectors = np.linalg.eig(self.cov_mat_target)
        # applying the transform to get the correct ellipse for our given covariance matrix
        ellipse = np.dot(np.dot(cov_eigenvectors, np.diag(cov_eigenvalues) ** (1/2)), circle.T).T
        # initialize plot and add ellipse
        return ax.plot(ellipse[:, 0], ellipse[:, 1])

    def make_circle(self):
        def unit_circle_values(t, r):
            return [r * math.cos(t), r*math.sin(t)]
        circle = np.zeros((200,2))
        for i in range(200):
            r = 2 # radius 2 implies that we draw the circle two standard deviations out from the center of a standard bivariate normal
            t = 2 * math.pi * (i / 200)
            circle[i, :] = unit_circle_values(t, r)
        return circle

    def plot_sample(self, annotate = False, save = False, show_data = True, trace_sample_path = False, alpha = 0.2):
        # initialize figure
        fig, ax = plt.subplots()
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])

        # plot ellipse into figure
        self.plot_ellipse(ax)
        # add sample to figure
        if trace_sample_path:
            ax.plot(self.sample[:,0], self.sample[:, 1], alpha = 0.5, color = 'black')
        if show_data:
            ax.plot(self.sample[:, 0], self.sample[:, 1], 'b.', alpha = alpha)

        if annotate:
            # annotate with parameters
            props = dict(boxstyle='square', facecolor='white', alpha=0.5)

            # defined text for annotation
            my_text = 'corr: ' + format(self.cov_mat_target[0, 1], '.2f') + '\n' + 'n: ' + str(self.n_samples)
            my_text += '\n' + 'n_eff: ' + str(int(self.effective_n))
            # place a text box in upper left in axes coords
            ax.text(0.05, 0.95, my_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
            ax.set_title("Gibbs Sample")
        # show plot
        if save:
            plt.savefig('gibbs_sample_biv_gauss_sample_1overcorr_' + format(1 / self.cov_mat_target[0,1], '.1f') + '.png')
        plt.show()

    def plot_autocorrelation(self, n_param = 0, save = False):
        # get autocorrelation by parameter
        sample_acf = sm.tsa.stattools.acf(self.sample[:, n_param])
        lags = np.arange(1, len(sample_acf), 1)
        fig, ax = plt.subplots()

        # annotate with parameters
        props = dict(boxstyle='square', facecolor='white', alpha=0.5)

        # defined text for annotation
        my_text = 'corr: ' + format(self.cov_mat_target[0, 1], '.2f') + '\n' + 'n: ' + str(self.n_samples)
        my_text += '\n' + 'n_eff: ' + str(int(self.effective_n))
        # place a text box in upper left in axes coords
        ax.text(0.05, 0.45, my_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        ax.set_title("Gibbs Autocorrelation")

        ax.bar(lags, sample_acf[1:], color = 'b')
        ax.set_ylim([-1, 1])

        if save:
            plt.savefig('gibbs_sample_biv_gauss_autocorr_1overcorr_' + format(1 / self.cov_mat_target[0,1], '.1f') + '_vratio_' + format(1 / self.cov_mat_target[0,0], '.1f') + '.png')
        plt.show()

class bivariate_normal_hmc(object):
    def __init__(self,
                 covariance_matrix_target = (np.matrix([[1, 0], [0, 1]])),
                 init = [0, 0],
                 n_samples = 200,
                 L = 25,
                 epsilon_range = [0.20, 0.25]):

        self.cov_mat_target = covariance_matrix_target
        self.n_samples = n_samples
        self.init = init
        self.epsilon_range = epsilon_range
        self.L = L
        self.sample = 'No sample yet, run the sampler first'
        self.acceptance_rate = 'No sample yet, run the sampler first'
        self.effective_n = 'No sample yet, run the sampler first'
        self.type_of_sampler = 'hmc'
        self.grad_U = nd.Jacobian(self.U)
        self.trace_mode = False
        self.trace = 'No trace yet, run trace_hmc_quantities()'

    def U(self, x):
        # U defines the negative log likelihood (ignoring any constant factors)
        cov_mat_inv = np.linalg.inv(self.cov_mat_target)
        # print(np.array(x).shape)
        return (1/2) * np.dot(np.dot(np.transpose(x), cov_mat_inv), x)

    def hmc_step(self, current_q):
        # in case we want to trace momentum and position within leapfrog steps
        # we initialize a storage matrix
        if self.trace_mode:
            trace_step = np.zeros((5,self.L))

        # store beginning position
        q = current_q.copy()

        # sample starting momentum
        p = np.matrix(np.random.multivariate_normal(np.zeros(len(self.init)), np.linalg.inv(np.diag(np.diag(self.cov_mat_target))), 1)).T # LAST CHANGE
        current_p = p.copy()

        # pick random epsilon from specified range
        if len(self.epsilon_range) == 1:
            epsilon = self.epsilon_range[0]
        else:
            epsilon = np.random.uniform(self.epsilon_range[0], self.epsilon_range[1], 1)[0]

        #print('p:', p)
        #print('q:', q)
        #print('problematic q update...: ', np.dot(p.T, np.diag(np.diag(self.cov_mat_target))))
        #print('type of problematic update: ', np.dot(p.T, np.diag(np.diag(self.cov_mat_target))).shape)

        # run leap-frog steps
        for i in range(self.L):
            p = p - ((epsilon / 2) * (self.grad_U(q).T))
            q = q + (epsilon * np.dot(p.T, np.diag(np.diag(self.cov_mat_target)))).T
            p = p - ((epsilon / 2) * (self.grad_U(q).T))

            if self.trace_mode:
                hamiltonian = self.U(q) + np.dot(np.dot(p.T, np.diag(np.diag(self.cov_mat_target))), p) / 2
                trace_step[0, i] = hamiltonian
                trace_step[1:3, i] = list(q)
                trace_step[3:5, i] = list(p)
                print('Traced', i, 'of', self.L, 'steps')

        # negate p (for mathematical consistency with respect to reversibility of the chain)
        p = - p

        # evaluate potential energy and kinetic energy at start and end of trajectory
        current_U = self.U(current_q)
        current_K = np.dot(np.dot(current_p.T, np.diag(np.diag(self.cov_mat_target))), current_p) / 2
        proposed_U = self.U(q)
        proposed_K = np.dot(np.dot(p.T, np.diag(np.diag(self.cov_mat_target))), p) / 2

        # Accept or reject the state at end of trajectory, returning either
        # the position at the end of trajectory or the initial position

        if self.trace_mode:
            self.trace = trace_step
        else:
            if np.log(np.random.uniform(0,1,1)) < (current_U - proposed_U + current_K - proposed_K):
                return np.matrix.tolist(q.T)[0], 1 # we accepted the proposal
            else:
                return np.matrix.tolist(current_q.T)[0], 0 # we rejected the proposal

    def run_sampler(self, feedback_after_n_samples = 20):
        self.sample = np.zeros((self.n_samples, len(self.init) + 1))
        self.sample[0, :len(self.init)] = self.init

        for i in range(1, self.n_samples, 1):
            self.sample[i,:len(self.init)], self.sample[i, len(self.init)] = self.hmc_step(np.matrix(self.sample[i - 1, :len(self.init)]).T)
            if feedback_after_n_samples != 0:
                if i % feedback_after_n_samples == 0 and i > 0:
                    print(i,'of', self.n_samples, 'samples drawn')
        print('sampler finished')
        self.acceptance_rate = sum(self.sample[:,2]) / len(self.sample[:,2])
        self.effective_n = self.get_effective_sample_size()

    def trace_hmc_quantities(self):
        self.trace_mode = True
        cur_init = np.matrix(np.random.multivariate_normal(np.zeros(len(self.init)), self.cov_mat_target, 1)).T
        self.hmc_step(cur_init)
        self.trace_mode = False

    def get_effective_sample_size(self):
        autocorrelation_times = list()
        for i in range(len(self.sample[0,:]) - 1):
            sample_acf = sm.tsa.stattools.acf(self.sample[:, i])
            tmp_autocorrelation_time = 1
            for k in sample_acf[1:]: # starting from one since lag-zero doesn't make sense
                if abs(k) > 0.0999:
                    tmp_autocorrelation_time += 2 * abs(k)
                else:
                    break
            autocorrelation_times.append(tmp_autocorrelation_time)
        effective_n_list = self.n_samples / np.array(autocorrelation_times)
        return min(effective_n_list)

    def plot_ellipse(self, ax, transform_to_apply_to_circle, color = 'black'):
        circle = self.make_circle()
        eigenvalues, eigenvectors = np.linalg.eig(transform_to_apply_to_circle)
        # applying the transform to get the correct ellipse for our given covariance matrix
        ellipse = np.dot(np.dot(eigenvectors, np.diag(eigenvalues) ** (1/2)), circle.T).T
        # initialize plot and add ellipse
        return ax.plot(ellipse[:, 0], ellipse[:, 1], color = color)

    def make_circle(self):
        def unit_circle_values(t, r):
            return [r * math.cos(t), r*math.sin(t)]
        circle = np.zeros((200,2))
        for i in range(200):
            r = 2 # radius 2 implies that we draw the circle two standard deviations out from the center of a standard bivariate normal
            t = 2 * math.pi * (i / 200)
            circle[i, :] = unit_circle_values(t, r)
        return circle

    def plot_sample(self, annotate = False, save = False, alpha = 0.2):
        # initialize figure
        fig, ax = plt.subplots()
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])

        # plot ellipse into figure
        self.plot_ellipse(ax, self.cov_mat_target)

        # add sample to figure
        ax.plot(self.sample[:, 0], self.sample[:, 1], 'b.', alpha = alpha)

        if annotate:
            # annotate with parameters
            props = dict(boxstyle='square', facecolor='white', alpha=0.5)

            # defined text for annotation
            my_text = 'cov: ' + format(self.cov_mat_target[0, 1], '.3f') + '\n' + 'n: ' + str(self.n_samples) + '\n' + 'L: ' + str(self.L) + '\n' + 'e: ' + str(self.epsilon_range[0]) + '-' + str(self.epsilon_range[1]) + '\n'
            my_text += 'acc: ' + format(self.acceptance_rate, '.2f') + '\n' + 'n_eff: ' + str(int(self.effective_n))
            # place a text box in upper left in axes coords
            ax.text(0.05, 0.95, my_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
            ax.set_title("HMC Sample")
        if save:
            plt.savefig('hmc_sample_biv_gauss_sample_vratio_' + format(1 / self.cov_mat_target[0,0], '.1f') + '.png')
        # show plot
        plt.show()

    def plot_autocorrelation(self, n_param = 0, save = False):
        # get autocorrelation by parameter
        sample_acf = sm.tsa.stattools.acf(self.sample[:, n_param])
        lags = np.arange(1, len(sample_acf), 1)
        fig, ax = plt.subplots()

        # annotate with parameters
        props = dict(boxstyle='square', facecolor='white', alpha=0.5)

        # defined text for annotation
        my_text = 'vratio: ' + format(1 / self.cov_mat_target[0, 1], '.1f') + '\n' + 'n: ' + str(self.n_samples) + '\n' + 'L: ' + str(self.L) + '\n'
        if len(self.epsilon_range) == 1:
            my_text += 'e: ' + str(self.epsilon_range[0]) + '\n'
        else:
            my_text += 'e: ' + str(self.epsilon_range[0]) + '-' + str(self.epsilon_range[1]) + '\n'

        my_text += 'n_eff: ' +  str(int(self.effective_n))
        # place a text box in upper left in axes coords
        ax.text(0.05, 0.45, my_text, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        ax.set_title("HMC Autocorrelation")

        ax.bar(lags, sample_acf[1:], color = 'b')
        ax.set_ylim([-1, 1])

        if save:
            plt.savefig('hmc_autocorr_biv_gauss_sample_vratio_' + format(1 / self.cov_mat_target[0,0], '.1f') + '.png')
        plt.show()

    def plot_coordinates(self, save = False, annotate = False):
        fig, ax = plt.subplots()
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        #plt.xlim(-3, 3)
        #plt.ylim(-3, 3)

        # annotate with parameters
        if annotate:
            props = dict(boxstyle='square', facecolor='white', alpha=0.5)

            # defined text for annotation
            my_text = 'cov: ' + format(self.cov_mat_target[0, 1], '.3f') + '\n' + 'n: ' + str(self.n_samples) + '\n' + 'L: ' + str(self.L) + '\n' + 'e: ' + str(self.epsilon_range[0]) + '-' + str(self.epsilon_range[1])
            # place a text box in upper left in axes coords
            ax.text(0.05, 0.95, my_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        ax.set_title("HMC Leap Frog Coordinates")
        self.plot_ellipse(ax, self.cov_mat_target)

        for i in range(len(self.trace[1, :]) - 1):
            ax.plot(self.trace[1, i:i + 2],
                   self.trace[2, i:i + 2],
                   '-.',
                   color = 'green',
                   alpha = i / len(self.trace[1,:]))

        if save:
            if self.cov_mat_target[0, 1] == 0:
                one_over_cov = 'inf'
            else:
                one_over_cov = str(int(1 / self.cov_mat_target[0,1]))
            plt.savefig('hmc_coordinates_biv_gauss_sample_vratio_' + format(1 / self.cov_mat_target[0,0], '.1f') + '1overcov_' + one_over_cov + '.png')

        plt.show()

    def plot_momentum(self, save = False, annotate = False):
        fig, ax = plt.subplots()
        #ax.set_xlim([-3, 3])
        #ax.set_ylim([-3, 3])

        if annotate:
            # annotate with parameters
            props = dict(boxstyle='square', facecolor='white', alpha=0.5)
            ax.axis('equal')

            # defined text for annotation
            my_text = 'cov: ' + format(self.cov_mat_target[0, 1], '.3f') + '\n' + 'n: ' + str(self.n_samples) + '\n' + 'L: ' + str(self.L) + '\n' + 'e: ' + str(self.epsilon_range[0]) + '-' + str(self.epsilon_range[1])
            # place a text box in upper left in axes coords
            ax.text(0.05, 0.95, my_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        ax.set_title("HMC Leap Frog Momentum")

        self.plot_ellipse(ax, np.linalg.inv(np.diag(np.diag(self.cov_mat_target))))
        for i in range(len(self.trace[1, :]) - 1):
            ax.plot(self.trace[3, i:i + 2],
                   self.trace[4, i:i + 2],
                   '-.',
                   color = 'blue',
                   alpha = i / len(self.trace[1,:]))

        if save:
            if self.cov_mat_target[0, 1] == 0:
                one_over_cov = 'inf'
            else:
                one_over_cov = str(int(1 / self.cov_mat_target[0,1]))
            plt.savefig('hmc_momentum_biv_gauss_sample_vratio_' + format(1 / self.cov_mat_target[0,0], '.1f') + '1overcov_' + one_over_cov + '.png')
        plt.show()

    def plot_hamiltonian(self, save = False, annotate = False):
        fig, ax = plt.subplots()
        l_steps = np.arange(1, len(self.trace[0,:]) + 1, 1)

        if annotate:
            # annotate with parameters
            props = dict(boxstyle='square', facecolor='white', alpha=0.5)

            # defined text for annotation
            my_text = 'cov: ' + format(self.cov_mat_target[0, 1], '.3f') + '\n' + 'n: ' + str(self.n_samples) + '\n' + 'L: ' + str(self.L) + '\n' + 'e: ' + str(self.epsilon_range[0]) + '-' + str(self.epsilon_range[1])
            # place a text box in upper left in axes coords
            ax.text(0.05, 0.95, my_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        ax.set_title("HMC Leap Frog Hamiltonian")

        for i in range(len(self.trace[1, :]) - 1):
            ax.plot(l_steps[i:i + 2],
                   self.trace[0, i:i + 2],
                   '-.',
                   color = 'blue',
                   alpha = i / len(self.trace[0, :]))

        if save:
            if self.cov_mat_target[0, 1] == 0:
                one_over_cov = 'inf'
            else:
                one_over_cov = str(int(1 / self.cov_mat_target[0,1]))

            plt.savefig('hmc_hamiltonian_biv_gauss_sample_vratio_' + format(1 / self.cov_mat_target[0,0], '.1f') + '1overcov_' + one_over_cov + '.png')
        plt.show()


# class bivariate_normal_hmc(object):
#     def __init__(self,
#                  covariance_matrix_target = (np.matrix([[1, 0], [0, 1]])),
#                  init = [0, 0],
#                  n_samples = 200,
#                  L = 25,
#                  epsilon_range = [0.20, 0.25]):
#
#         self.cov_mat_target = covariance_matrix_target
#         self.n_samples = n_samples
#         self.init = init
#         self.epsilon_range = epsilon_range
#         self.L = L
#         self.sample = 'No sample yet, run the sampler first'
#         self.acceptance_rate = 'No sample yet, run the sampler first'
#         self.type_of_sampler = 'hmc'
#         self.grad_U = nd.Jacobian(self.U)
#         self.trace_mode = False
#         self.trace = 'No trace yet, run trace_hmc_quantities()'
#
#     def U(self, x):
#         # U defines the negative log likelihood (ignoring any constant factors)
#         cov_mat_inv = np.linalg.inv(self.cov_mat_target)
#         # print(np.array(x).shape)
#         return (1/2) * np.dot(np.dot(np.transpose(x), cov_mat_inv), x)
#
#     def hmc_step(self, current_q):
#         # in case we want to trace momentum and position within leapfrog steps
#         # we initialize a storage matrix
#         if self.trace_mode:
#             trace_step = np.zeros((5,self.L))
#
#         # store beginning position
#         q = current_q.copy()
#
#         # sample starting momentum
#         p = np.matrix(np.random.multivariate_normal(np.zeros(len(self.init)), np.diag(np.ones(len(self.init))), 1)).T
#         current_p = p.copy()
#
#         # pick random epsilon from specified range
#         if len(self.epsilon_range) == 1:
#             epsilon = self.epsilon_range
#         else:
#             epsilon = np.random.uniform(self.epsilon_range[0], self.epsilon_range[1], 1)[0]
#
#         # run leap-frog steps
#         for i in range(self.L):
#             p = p - ((epsilon / 2) * (self.grad_U(q).T))
#             q = q + (epsilon * p)
#             p = p - ((epsilon / 2) * (self.grad_U(q).T))
#
#             if self.trace_mode:
#                 hamiltonian = self.U(q) + np.sum(np.power(p,2)) / 2
#                 trace_step[0, i] = hamiltonian
#                 trace_step[1:3, i] = list(q)
#                 trace_step[3:5, i] = list(p)
#                 print('Traced', i, 'of', self.L, 'steps')
#
#         # negate p (for mathematical consistency with respect to reversibility of the chain)
#         p = -p
#
#         # evaluate potential energy and kinetic energy at start and end of trajectory
#         current_U = self.U(current_q)
#         current_K = np.sum(np.power(current_p, 2)) / 2
#         proposed_U = self.U(q)
#         proposed_K = np.sum(np.power(p, 2)) / 2
#
#         # Accept or reject the state at end of trajectory, returning either
#         # the position at the end of trajectory or the initial position
#
#         if self.trace_mode:
#             self.trace = trace_step
#         else:
#             if np.log(np.random.uniform(0,1,1)) < (current_U - proposed_U + current_K - proposed_K):
#                 return np.matrix.tolist(q.T)[0], 1 # we accepted the proposal
#             else:
#                 return np.matrix.tolist(current_q.T)[0], 0 # we rejected the proposal
#
#     def run_sampler(self):
#         self.sample = np.zeros((self.n_samples, len(self.init) + 1))
#         self.sample[0, :len(self.init)] = self.init
#
#         for i in range(1, self.n_samples, 1):
#             self.sample[i,:len(self.init)], self.sample[i, len(self.init)] = self.hmc_step(np.matrix(self.sample[i - 1, :len(self.init)]).T)
#             if i % 10 == 0:
#                 print(i,'of', self.n_samples, 'samples drawn')
#         self.acceptance_rate = sum(self.sample[:,2]) / len(self.sample[:,2])
#
#     def trace_hmc_quantities(self):
#         self.trace_mode = True
#         cur_init = np.matrix(np.random.multivariate_normal(np.zeros(len(self.init)), np.diag(np.ones(len(self.init))), 1)).T
#         self.hmc_step(cur_init)
#         self.trace_mode = False
#
#     def plot_ellipse(self, ax):
#         circle = self.make_circle()
#         cov_eigenvalues, cov_eigenvectors = np.linalg.eig(self.cov_mat_target)
#         # applying the transform to get the correct ellipse for our given covariance matrix
#         ellipse = np.dot(np.dot(cov_eigenvectors, np.diag(cov_eigenvalues) ** (1/2)), circle.T).T
#         # initialize plot and add ellipse
#         return ax.plot(ellipse[:, 0], ellipse[:, 1])
#
#     def make_circle(self):
#         def unit_circle_values(t, r):
#             return [r * math.cos(t), r*math.sin(t)]
#         circle = np.zeros((200,2))
#         for i in range(200):
#             r = 2 # radius 2 implies that we draw the circle two standard deviations out from the center of a standard bivariate normal
#             t = 2 * math.pi * (i / 200)
#             circle[i, :] = unit_circle_values(t, r)
#         return circle
#
#     def plot_sample(self):
#         # initialize figure
#         fig, ax = plt.subplots()
#         ax.set_xlim([-3, 3])
#         ax.set_ylim([-3, 3])
#
#         # plot ellipse into figure
#         self.plot_ellipse(ax)
#
#         # add sample to figure
#         ax.plot(self.sample[:, 0], self.sample[:, 1], 'b.', alpha = 0.2)
#
#         # annotate with parameters
#         props = dict(boxstyle='square', facecolor='white', alpha=0.5)
#
#         # defined text for annotation
#         my_text = 'corr: ' + str(self.cov_mat_target[0, 1]) + '\n' + 'n: ' + str(self.n_samples) + '\n' + 'L: ' + str(self.L) + '\n' + 'e: ' + str(self.epsilon_range[0]) + '-' + str(self.epsilon_range[1]) + '\n' + 'acc: ' + format(self.acceptance_rate, '.2f')
#         # place a text box in upper left in axes coords
#         ax.text(0.05, 0.95, my_text, transform=ax.transAxes, fontsize=14,
#             verticalalignment='top', bbox=props)
#         ax.set_title("HMC Sample")
#
#         # show plot
#         plt.show()
#
#     def plot_autocorrelation(self, n_param):
#         # get autocorrelation by parameter
#         sample_acf = sm.tsa.stattools.acf(self.sample[:, n_param])
#         lags = np.arange(1, len(sample_acf), 1)
#         fig, ax = plt.subplots()
#
#         # annotate with parameters
#         props = dict(boxstyle='square', facecolor='white', alpha=0.5)
#
#         # defined text for annotation
#         my_text = 'corr: ' + str(self.cov_mat_target[0, 1]) + '\n' + 'n: ' + str(self.n_samples) + '\n' + 'L: ' + str(self.L) + '\n' + 'e: ' + str(self.epsilon_range[0]) + '-' + str(self.epsilon_range[1])
#         # place a text box in upper left in axes coords
#         ax.text(0.05, 0.95, my_text, transform=ax.transAxes, fontsize=14,
#             verticalalignment='top', bbox=props)
#         ax.set_title("HMC Autocorrelation")
#
#         ax.bar(lags, sample_acf[1:], color = 'b')
#         ax.set_ylim([-1, 1])
#         plt.show()
#
#     def plot_coordinates(self):
#         fig, ax = plt.subplots()
#         ax.set_xlim([-3, 3])
#         ax.set_ylim([-3, 3])
#         #plt.xlim(-3, 3)
#         #plt.ylim(-3, 3)
#
#         # annotate with parameters
#         props = dict(boxstyle='square', facecolor='white', alpha=0.5)
#
#         # defined text for annotation
#         my_text = 'corr: ' + str(self.cov_mat_target[0, 1]) + '\n' + 'n: ' + str(self.n_samples) + '\n' + 'L: ' + str(self.L) + '\n' + 'e: ' + str(self.epsilon_range[0]) + '-' + str(self.epsilon_range[1])
#         # place a text box in upper left in axes coords
#         ax.text(0.05, 0.95, my_text, transform=ax.transAxes, fontsize=14,
#             verticalalignment='top', bbox=props)
#         ax.set_title("HMC Leap Frog Coordinates")
#
#
#         self.plot_ellipse(ax)
#         for i in range(len(self.trace[1, :]) - 1):
#             ax.plot(self.trace[1, i:i + 2],
#                    self.trace[2, i:i + 2],
#                    '-.',
#                    color = 'green',
#                    alpha = i / len(self.trace[1,:]))
#         plt.show()
#
#     def plot_momentum(self):
#         fig, ax = plt.subplots()
#         plt.xlim(-3, 3)
#         plt.ylim(-3, 3)
#         circle = self.make_circle()
#
#         # annotate with parameters
#         props = dict(boxstyle='square', facecolor='white', alpha=0.5)
#
#         # defined text for annotation
#         my_text = 'corr: ' + str(self.cov_mat_target[0, 1]) + '\n' + 'n: ' + str(self.n_samples) + '\n' + 'L: ' + str(self.L) + '\n' + 'e: ' + str(self.epsilon_range[0]) + '-' + str(self.epsilon_range[1])
#         # place a text box in upper left in axes coords
#         ax.text(0.05, 0.95, my_text, transform=ax.transAxes, fontsize=14,
#             verticalalignment='top', bbox=props)
#         ax.set_title("HMC Leap Frog Momentum")
#
#         ax.plot(circle[:, 0], circle[:, 1], color = 'black')
#         for i in range(len(self.trace[1, :]) - 1):
#             ax.plot(self.trace[3, i:i + 2],
#                    self.trace[4, i:i + 2],
#                    '-.',
#                    color = 'blue',
#                    alpha = i / len(self.trace[1,:]))
#         plt.show()
#
#     def plot_hamiltonian(self):
#         fig, ax = plt.subplots()
#         l_steps = np.arange(1, len(self.trace[0,:]) + 1, 1)
#
#         # annotate with parameters
#         props = dict(boxstyle='square', facecolor='white', alpha=0.5)
#
#         # defined text for annotation
#         my_text = 'corr: ' + str(self.cov_mat_target[0, 1]) + '\n' + 'n: ' + str(self.n_samples) + '\n' + 'L: ' + str(self.L) + '\n' + 'e: ' + str(self.epsilon_range[0]) + '-' + str(self.epsilon_range[1])
#         # place a text box in upper left in axes coords
#         ax.text(0.05, 0.95, my_text, transform=ax.transAxes, fontsize=14,
#             verticalalignment='top', bbox=props)
#         ax.set_title("HMC Leap Frog Hamiltonian")
#
#         for i in range(len(self.trace[1, :]) - 1):
#             ax.plot(l_steps[i:i + 2],
#                    self.trace[0, i:i + 2],
#                    '-.',
#                    color = 'blue',
#                    alpha = i / len(self.trace[0, :]))
#         plt.show()
