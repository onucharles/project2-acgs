import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def get_experiment_folder(exp_id):
    return Path('output') / exp_id

def plot_learning_curves(exp_path):
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
    print('best val ppl is {}. train ppl is {} at {} epoch'.format(best_val, train_ppls[best_val_idx], x[best_val_idx]))

    plt.figure()

    # plot perplexity on epochs
    plt.subplot(1,2,1)
    plt.plot(x, train_ppls, label='training')
    plt.plot(x, val_ppls, label='validation')
    plt.ylabel('perplexity', fontsize=13)
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
    plt.yscale('symlog')
    plt.xlabel('wall clock time (s)', fontsize=13)
    plt.title('(b)', fontsize=13)
    plt.grid()
    plt.legend(fontsize=13)

    plt.show()

# NEXT IMPLEMENT THIS FUNCTION
def plot_val_from_folder(source_dir):
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
    ax0.set_yscale('symlog')
    ax0.set_xlabel('epochs', fontsize=13)
    ax0.set_title('(a)', fontsize=13)
    ax0.grid()
    ax0.legend(fontsize=13)

    ax1.set_ylabel('perplexity', fontsize=13)
    ax1.set_yscale('symlog')
    ax1.set_xlabel('wall clock time (s)', fontsize=13)
    ax1.set_title('(b)', fontsize=13)
    ax1.grid()
    ax1.legend(fontsize=13)

    plt.show()

def main():
    # exp1 RNN,
    # plot_learning_curves(get_experiment_folder(''))
    # exp2 RNN,
    # plot_learning_curves(get_experiment_folder(''))
    # exp3 RNN,
    # plot_learning_curves(get_experiment_folder(''))
    # exp4 RNN,
    # plot_learning_curves(get_experiment_folder(''))
    # exp5 RNN,
    # plot_learning_curves(get_experiment_folder(''))
    # exp6 RNN,
    # plot_learning_curves(get_experiment_folder(''))

    # exp1 GRU,
    # plot_learning_curves(get_experiment_folder(''))
    # exp2 GRU,
    # plot_learning_curves(get_experiment_folder(''))
    # exp3 GRU,
    # plot_learning_curves(get_experiment_folder(''))
    # exp4 GRU,
    # plot_learning_curves(get_experiment_folder(''))
    # exp5 GRU,
    # plot_learning_curves(get_experiment_folder(''))
    # exp6 GRU,
    # plot_learning_curves(get_experiment_folder(''))

    # exp1 TRANSFORMER, SGD
    # plot_learning_curves(get_experiment_folder('c25de3481cba4e15b568a6b9f3b64256'))
    # exp2 TRANSFORMER, SGD_LR_SCHEDULE
    # plot_learning_curves(get_experiment_folder('ff9e25a20ed94bbe9161126d6b7348ee'))
    # exp3 TRANSFORMER, ADAM
    # plot_learning_curves(get_experiment_folder('baa5d573c5fb45b882521c68a3886c9f'))
    # exp4 TRANSFORMER, ADAM lr=0.0001
    # plot_learning_curves(get_experiment_folder('0ea3a6f51e614a6b9ee1fa913e70e083'))
    # exp5 TRANSFORMER, ADAM lr=0.0001, seq_len=45
    # plot_learning_curves(get_experiment_folder('00439eb99bf74f8d9cfb1a88835fd69e'))
    # exp6 TRANSFORMER, ADAM lr=0.0001, dropout-keep=0.8
    # plot_learning_curves(get_experiment_folder('699d09435c3043a8bd19ef351b4696a3'))

    # ---------Optimizers---------
    #adam folder
    #sgd fodler
    #sgd-lr-schedule folder

    #---------Architectures---------
    #rnn folder
    #gru folder
    #transformer folder
    plot_val_from_folder(get_experiment_folder('transformer'))

if __name__== '__main__':
    main()
