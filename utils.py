import numpy as np
import os
import pandas as pd


class Tee:
    def __init__(self, fname, stream, mode="a+"):
        self.stream = stream
        self.file = open(fname, mode)

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


def flatten_dictionary_for_wandb(args):
    """
    Small script to keep a clean Wandb config file by having easy access to the hyperparameters
    """
    # Check if 'SMA' is a key in the dictionary and if its value is a dictionary
    if 'SMA' in args:
        # Update the original dictionary with the values from args['SMA']
        args.update(args['SMA'])
        # Remove the 'SMA' key from the dictionary
        del args['SMA']
    return args


def list_to_matrix(lst, mode):
    if mode != 'binary':

        K = int(np.sqrt(len(lst)))
        return np.array(lst).reshape(K, K)
    else:
        K = int(len(lst)) // 2
        return np.array(lst).reshape(2, K)


def zero_diagonal(M):
    """Sets the diagonal of the matrix to zero"""
    M_copy = np.copy(M)
    np.fill_diagonal(M_copy, 0)
    return M_copy


def results_to_log_dict(result, mode):
    """ Computes the mean, worst, minor, major and relative accuracy from a list of accuracies"""
    log_dict = {'lr': result['lr']}
    for acc in ['acc_tr', 'acc_va', 'acc_te']:
        if acc in result:
            acc_mat = list_to_matrix(result[acc], mode) * 100
            K = acc_mat.shape[0]
            log_dict[f'mean_grp_{acc}'] = np.mean(acc_mat)
            log_dict[f'worst_grp_{acc}'] = np.min(acc_mat)
            if mode != 'binary':
                minor_acc = zero_diagonal(acc_mat).sum() / (K * (K - 1))
                major_acc = np.trace(acc_mat) / K
            else:
                minor_acc = zero_diagonal(acc_mat).sum() / (2 * (K - 1))
                major_acc = np.trace(acc_mat) / 2
            log_dict[f'minor_grp_{acc}'] = minor_acc
            log_dict[f'major_grp_{acc}'] = major_acc
            log_dict[f'relative_grp_{acc}'] = minor_acc / major_acc

    return log_dict
