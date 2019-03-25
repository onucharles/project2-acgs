import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def get_experiment_folder(exp_id):
    return Path('output') / exp_id

def plot_learning_curves(exp_path, y_log_scale=True):
    lc_path = exp_path / 'learning_curves.npy'

    print('Plot learning curves -')
    print('Loading logs saved at: ', lc_path)
    x = np.load(lc_path)[()]

    train_ppls = np.array(x['train_ppls'])
    val_ppls = np.array(x['val_ppls'])
    times = np.array(x['times'])
    print('train logs length ', len(train_ppls))
    print('valid logs length ', len(val_ppls))
    print('time length ', len(times))

    x = np.arange(len(train_ppls)) + 1
    best_val = np.min(val_ppls)
    best_val_idx = val_ppls == best_val
    print('train ppl is {}. best val ppl is {}. at {} epoch'.format(train_ppls[best_val_idx], best_val, x[best_val_idx]))

    plt.figure()

    # plot perplexity on epochs
    plt.subplot(1,2,1)
    plt.plot(x, train_ppls, label='training')
    plt.plot(x, val_ppls, label='validation')
    plt.ylabel('perplexity', fontsize=13)
    if y_log_scale:
        plt.yscale('symlog')
    plt.xlabel('epochs', fontsize=13)
    plt.title('(a)', fontsize=13)
    plt.grid()
    plt.legend(fontsize=13)

    # plot perplexity on clock time.
    plt.subplot(1, 2, 2)
    wall_time = np.cumsum(times)
    plt.plot(wall_time, train_ppls, label='training')
    plt.plot(wall_time, val_ppls, label='validation')
    plt.ylabel('perplexity', fontsize=13)
    if y_log_scale:
        plt.yscale('symlog')
    plt.xlabel('wall clock time (s)', fontsize=13)
    plt.title('(b)', fontsize=13)
    plt.grid()
    plt.legend(fontsize=13)

    plt.show()

# NEXT IMPLEMENT THIS FUNCTION
def plot_val_from_folder(source_dir, y_log_scale=True):
    """
    We assume that only files with extension '.npy' are in
    specified directory.
    """

    if not os.path.exists(source_dir) or not os.path.isdir(source_dir):
        raise ValueError('source_dir does not exist or is not a directory')

    n_epochs = 30
    plt.figure()
    ax0 = plt.subplot(1, 2, 1)
    ax1 = plt.subplot(1, 2, 2)

    for file in os.listdir(source_dir):
        if not file.endswith('.npy'):
            continue
        lc_path = os.path.join(source_dir, file)
        print('Loading logs saved at: ', lc_path)
        lc = np.load(lc_path)[()]
        exp_id = file.split('.')[0]

        val_ppls = np.array(lc['val_ppls'])
        times = np.array(lc['times'])
        val_ppls = val_ppls[:n_epochs]
        times = times[:n_epochs]
        print('valid logs length ', len(val_ppls))
        print('time length ', len(times))

        x = np.arange(n_epochs) + 1
        best_val = np.min(val_ppls)
        best_val_idx = val_ppls == best_val
        print('best val ppl is {} at {} epoch'.format(best_val, x[best_val_idx]))

        # plot perplexity on epochs
        ax0.plot(x, val_ppls, label=exp_id)

        # plot perplexity on clock time.
        wall_time = np.cumsum(times)
        print(times)
        print(wall_time)
        ax1.plot(wall_time, val_ppls, label=exp_id)


    ax0.set_ylabel('perplexity', fontsize=13)
    if y_log_scale:
        ax0.set_yscale('symlog')
    ax0.set_xlabel('epochs', fontsize=13)
    ax0.set_title('(a)', fontsize=13)
    ax0.grid()
    ax0.legend(fontsize=13)

    ax1.set_ylabel('perplexity', fontsize=13)
    if y_log_scale:
        ax1.set_yscale('symlog')
    ax1.set_xlabel('wall clock time (s)', fontsize=13)
    ax1.set_title('(b)', fontsize=13)
    ax1.grid()
    ax1.legend(fontsize=13)

    plt.show()

def main():
    # exp1 RNN,
    # plot_learning_curves(get_experiment_folder('e116883eee38438897b4f60708c4620c'), False)
    # exp2 RNN,
    # plot_learning_curves(get_experiment_folder('5899ae75b9544b598cb5d76a8c5644a3'), False)
    # exp3 RNN,
    # plot_learning_curves(get_experiment_folder('0540797611274b40965ac1f99114a040'), False)
    # exp4 RNN,
    # plot_learning_curves(get_experiment_folder('a3b6ba9b328d4081bf07a6081a28692d'), False)
    # exp5 RNN,
    # plot_learning_curves(get_experiment_folder('dea908eaa6c743f1a35a1364e1834406'), False)
    # exp6 RNN,
    # plot_learning_curves(get_experiment_folder('5759ebc5431d43498e7b8a2963da4d64'), False)

    # exp1 GRU,
    # plot_learning_curves(get_experiment_folder('8ccc927139a041ebb201dc57f3bf6aec'), False)
    # exp2 GRU,
    # plot_learning_curves(get_experiment_folder('2a1bc552523e4a90a684f6840a31397d'), False)
    # exp3 GRU,
    # plot_learning_curves(get_experiment_folder('084e720caa034fae8b8b8def08b9eddf'), False)
    # exp4 GRU,
    # plot_learning_curves(get_experiment_folder('3499412f136341bb903d6454583e9fcd'), False)
    # exp5 GRU,
    # plot_learning_curves(get_experiment_folder('1df7c617c02647029d6d3b53af8d3c3f'), False)
    # exp6 GRU,
    # plot_learning_curves(get_experiment_folder('gru-exp6'), False)

    # exp1 TRANSFORMER
    # plot_learning_curves(get_experiment_folder('e277e50cb1a64f639e8b9b7abff3b58b'))
    # exp2 TRANSFORMER,
    # plot_learning_curves(get_experiment_folder('3d97f80e062540c29dad0ba41c115fa3'))
    # exp3 TRANSFORMER,
    # plot_learning_curves(get_experiment_folder('fce42069e5c046e49477f158400d0ee5'))
    # exp4 TRANSFORMER,
    # plot_learning_curves(get_experiment_folder('b70bf0e5e13a425a91bc28b02dd81f7d'), False)
    # exp5 TRANSFORMER,
    # plot_learning_curves(get_experiment_folder('fa5ca509d4e6420abd00412b4f219f44'), False)
    # exp6 TRANSFORMER,
    # plot_learning_curves(get_experiment_folder('c765a5ee76bf4d2bbaa517b763ae441b'), False)

    # ---------Optimizers---------
    #adam folder
    # plot_val_from_folder(get_experiment_folder('adam'))
    #sgd fodler
    # plot_val_from_folder(get_experiment_folder('sgd'))
    #sgd-lr-schedule folder
    # plot_val_from_folder(get_experiment_folder('sgd-lr'))

    #---------Architectures---------
    #rnn folder
    # plot_val_from_folder(get_experiment_folder('rnn'))
    #gru folder
    # plot_val_from_folder(get_experiment_folder('gru'), False)
    #transformer folder
    plot_val_from_folder(get_experiment_folder('transformer'))

if __name__== '__main__':
    main()
