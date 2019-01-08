from __future__ import print_function
import json
import numpy as np
import sys


def forward(pi, A, B, O):
    """
    Forward algorithm

    Inputs:
    - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
    - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
    - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
    - O: A list of observation sequence (in terms of index, not the actual symbol)

    Returns:
    - alpha: A numpy array alpha[j, t-1] = P(Z_t = s_j, X_{1:t}=x_{1:t})
    """
    S = len(pi)
    N = len(O)
    alpha = np.zeros([S, N])
    ###################################################
    # Q3.1 Edit here

    alpha[:, 0] = pi * B[:, O[0]]
    for i in range(1, N):
        obs = B[:, O[i]]
        prev_alpha = alpha[:, i - 1]
        sum = np.dot(prev_alpha, A)
        alpha[:, i] = obs * sum

    ###################################################

    return alpha


def backward(pi, A, B, O):
    """
    Backward algorithm

    Inputs:
    - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
    - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
    - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
    - O: A list of observation sequence (in terms of index, not the actual symbol)

    Returns:
    - beta: A numpy array beta[j, t-1] = P(X_{t+1:N}=x_{t+1:N} | Z_t = s_j)
    """
    S = len(pi)
    N = len(O)
    beta = np.zeros([S, N])
    ###################################################
    # Q3.1 Edit here

    beta[:, N - 1] = 1
    for i in range(N - 2, -1, -1):
        obs = B[:, O[i + 1]]
        next_beta = beta[:, i + 1]
        temp = obs * next_beta
        beta[:, i] = np.dot(temp.T, A.T)

    ###################################################
    return beta


def seqprob_forward(alpha):
    """
    Total probability of observing the whole sequence using the forward messages

    Inputs:
    - alpha: A numpy array alpha[j, t-1] = P(Z_t = s_j, X_{1:t}=x_{1:t})

    Returns:
    - prob: A float number of P(X_{1:N}=O)
    """
    prob = 0
    N = alpha.shape[1]
    ###################################################
    # Q3.2 Edit here

    prob = np.sum(alpha[:, N - 1])

    ###################################################
    return prob


def seqprob_backward(beta, pi, B, O):
    """
    Total probability of observing the whole sequence using the backward messages

    Inputs:
    - beta: A numpy array beta: A numpy array beta[j, t-1] = P(X_{t+1:N}=x_{t+1:N} | Z_t = s_j)
    - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
    - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
    - O: A list of observation sequence
        (in terms of the observation index, not the actual symbol)

    Returns:
    - prob: A float number of P(X_{1:N}=O)
    """
    prob = 0
    ###################################################
    # Q3.2 Edit here

    emit = B[:, O[0]]
    beta_ = beta[:, 0]
    prob = np.sum(emit * pi * beta_)

    ###################################################
    return prob


def viterbi(pi, A, B, O):
    """
    Viterbi algorithm

    Inputs:
    - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
    - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
    - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
    - O: A list of observation sequence (in terms of index, not the actual symbol)

    Returns:
    - path: A list of the most likely hidden state path (in terms of the state index)
    """
    path = []
    S = len(pi)
    N = len(O)
    gamma = np.zeros([S, N], dtype=float)
    delta = np.zeros([S, N], dtype=np.int64)
    ###################################################
    # Q3.3 Edit here

    gamma[:, 0] = pi * B[:, O[0]]
    delta[:, 0] = -1
    for i in range(1, N):
        obs = B[:, O[i]]
        prev_gamma = gamma[:, i - 1]
        gamma[:, i] = obs * np.max((prev_gamma.reshape(S, 1) * A), axis=0)
        delta[:, i] = np.argmax((prev_gamma.reshape(S, 1) * A), axis=0)

    prev = np.argmax(gamma[:, N - 1], axis=0)
    path.append(prev)
    for i in range(N - 1, 0, -1):
        current = delta[prev, i]
        path.append(current)
        prev = current

    path.reverse()

    ###################################################

    return path


##### DO NOT MODIFY ANYTHING BELOW THIS ###################
def main():
    model_file = sys.argv[1]
    Osymbols = sys.argv[2]

    #### load data ####
    with open(model_file, 'r') as f:
        data = json.load(f)
    A = np.array(data['A'])
    B = np.array(data['B'])
    pi = np.array(data['pi'])
    #### observation symbols #####
    obs_symbols = data['observations']
    #### state symbols #####
    states_symbols = data['states']

    N = len(Osymbols)
    O = [obs_symbols[j] for j in Osymbols]

    alpha = forward(pi, A, B, O)
    beta = backward(pi, A, B, O)

    prob1 = seqprob_forward(alpha)
    prob2 = seqprob_backward(beta, pi, B, O)
    print(
        'Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

    viterbi_path = viterbi(pi, A, B, O)
    print('Viterbi best path is ')
    for j in viterbi_path:
        print(states_symbols[j], end=' ')


if __name__ == "__main__":
    main()
