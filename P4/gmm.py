import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures (Int)
            e : error tolerance (Float) 
            max_iter : maximum number of updates (Int)
            init : initialization of means and variance
                Can be 'random' or 'kmeans' 
            means : means of Gaussian mixtures (n_cluster X D numpy array)
            variances : variance of Gaussian mixtures (n_cluster X D X D numpy array) 
            pi_k : mixture probabilities of different component ((n_cluster,) size numpy array)
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

        self.estimation = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):

            kmeans_obj = KMeans(self.n_cluster, self.max_iter, self.e)
            means, membership, max_iter = kmeans_obj.fit(x)
            self.means = means
            self.variances = np.zeros((self.n_cluster, D, D))

            k_lens = np.zeros(self.n_cluster)

            for k in range(self.n_cluster):
                indices_for_k = np.where(membership == k)
                len_for_k = len(indices_for_k[0])
                k_lens[k] = len_for_k
                x_for_k = x[indices_for_k]
                means_for_k = means[k].reshape(1, D)
                temp = (x_for_k - means_for_k)
                variance_for_k = np.dot(temp.T, temp)
                variance_for_k = np.divide(variance_for_k, len_for_k)
                self.variances[k] = variance_for_k

            self.pi_k = k_lens / N

            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k (see P4.pdf)

            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception(
            #     'Implement initialization of variances, means, pi_k using k-means')
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):

            self.means = np.random.rand(self.n_cluster, D)
            identity_matrix = np.identity(D)
            self.variances = np.array(self.n_cluster * [identity_matrix])
            self.pi_k = [(1 / self.n_cluster) for i in range(self.n_cluster)]

            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - initialize variance to be identity and pi_k to be uniform

            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception(
            #     'Implement initialization of variances, means, pi_k randomly')
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        self.estimation = np.zeros((N, self.n_cluster))
        log_likelihood = self.compute_log_likelihood(x, self.means, self.variances, self.pi_k)
        iterations = 0

        while iterations <= self.max_iter:

            # E step
            estimation = self.estimation
            estimation_sum = np.sum(estimation, axis=1)
            estimation = np.divide(estimation, estimation_sum.reshape(N, 1))
            self.estimation = estimation

            # M step
            N_estimation = np.sum(self.estimation, axis=0)

            for k in range(self.n_cluster):
                if N_estimation[k] != 0:
                    self.means[k] = np.sum(np.multiply(np.reshape(self.estimation[:, k], (N, 1)), x), axis=0) / \
                                    N_estimation[k]

                    a = np.multiply(x - self.means[k], np.reshape(self.estimation[:, k], (N, 1))).T
                    b = x - self.means[k]
                    self.variances[k] = np.matmul(a, b) / N_estimation[k]

            self.pi_k = N_estimation / N
            log_likelihood_new = self.compute_log_likelihood(x, self.means, self.variances, self.pi_k)
            if np.abs(log_likelihood - log_likelihood_new) <= self.e:
                break
            else:
                log_likelihood = log_likelihood_new

            iterations += 1

        return iterations + 1

    # TODO
    # - comment/remove the exception
    # - Use EM to learn the means, variances, and pi_k and assign them to self
    # - Update until convergence or until you have made self.max_iter updates.
    # - Return the number of E/M-Steps executed (Int)
    # Hint: Try to separate E & M step for clarity
    # DONOT MODIFY CODE ABOVE THIS LINE
    # raise Exception('Implement fit function (filename: gmm.py)')
    # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        D = self.means.shape[1]
        samples = np.zeros([N, D])
        for i in range(0, N):
            values = np.random.multinomial(1, self.pi_k)
            values_max = np.argmax(values)
            samples[i] = np.random.multivariate_normal(self.means[values_max], self.variances[values_max])

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement sample function in gmm.py')
        # DONOT MODIFY CODE BELOW THIS LINE
        return samples

    def compute_log_likelihood(self, x, means=None, variances=None, pi_k=None):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'
        if means is None:
            means = self.means
        if variances is None:
            variances = self.variances
        if pi_k is None:
            pi_k = self.pi_k

        N, D = x.shape

        for k in range(self.n_cluster):
            pi_k_k = pi_k[k]
            variance_k = variances[k]

            while np.linalg.matrix_rank(variance_k) < D:
                variance_k += 0.001 * np.eye(D)

            mean_k = means[k].reshape(1, D)

            varinace_k_inverse = np.linalg.inv(variance_k)
            variance_k_mod = np.linalg.det(variance_k)
            two_pi_d = (2 * np.pi) ** D
            denom = np.sqrt(two_pi_d * variance_k_mod)
            num_by_denom = 1 / denom

            one = x - mean_k
            two = varinace_k_inverse
            final = (-1 / 2) * np.sum(np.multiply(np.dot(one, two), one), axis=1)
            final_exp = np.exp(final)
            final_exp *= num_by_denom

            self.estimation[:, k] = pi_k_k * final_exp

        sum = np.sum(self.estimation, axis=1)
        sum = np.log(sum)
        log_likelihood = np.sum(sum, axis=0)

        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood (Float)
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        # raise Exception('Implement compute_log_likelihood function in gmm.py')
        # DONOT MODIFY CODE BELOW THIS LINE
        return float(log_likelihood)

    class Gaussian_pdf():
        def __init__(self, mean, variance):
            self.mean = mean
            self.variance = variance
            self.c = None
            self.inv = None
            '''
                Input: 
                    Means: A 1 X D numpy array of the Gaussian mean
                    Variance: A D X D numpy array of the Gaussian covariance matrix
                Output: 
                    None: 
            '''
            mu, D = mean, mean.shape[0]
            cov = self.variance
            while np.linalg.matrix_rank(cov) < D:
                cov += 0.001 * np.identity(D)
            self.inv = np.linalg.inv(cov)
            self.c = (((2 * np.pi) ** len(mu)) * np.linalg.det(cov))

            # TODO
            # - comment/remove the exception
            # - Set self.inv equal to the inverse the variance matrix (after ensuring it is full rank - see P4.pdf)
            # - Set self.c equal to ((2pi)^D) * det(variance) (after ensuring the variance matrix is full rank)
            # Note you can call this class in compute_log_likelihood and fit
            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception('Impliment Guassian_pdf __init__')
            # DONOT MODIFY CODE BELOW THIS LINE

        def getLikelihood(self, x):
            '''
                Input:
                    x: a 1 X D numpy array representing a sample
                Output:
                    p: a numpy float, the likelihood sample x was generated by this Gaussian
                Hint:
                    p = e^(-0.5(x-mean)*(inv(variance))*(x-mean)'/sqrt(c))
                    where ' is transpose and * is matrix multiplication
            '''

            temp = (-1 / 2) * ((x - self.mean).dot(self.inv).dot((x - self.mean).T))
            p = np.exp(temp) / np.sqrt(self.c)

            # TODO
            # - Comment/remove the exception
            # - Calculate the likelihood of sample x generated by this Gaussian
            # Note: use the described implementation of a Gaussian to ensure compatibility with the solutions
            # DONOT MODIFY CODE ABOVE THIS LINE
            # raise Exception('Impliment Guassian_pdf getLikelihood')
            # DONOT MODIFY CODE BELOW THIS LINE
            return p
