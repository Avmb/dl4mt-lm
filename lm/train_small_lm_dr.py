import os
import sys
import argparse

from small_lm_dr import train

def main(job_id):
    parser = argparse.ArgumentParser(description='Train a language model.')
    parser.add_argument('model', help='model file')
    parser.add_argument('dictionary', help='dictionary file')
    parser.add_argument('train_dataset', help='training dataset')
    parser.add_argument('valid_dataset', help='validation dataset')
    parser.add_argument('-t', '--test_dataset', help='test dataset')
    parser.add_argument('-r', '--reload', help='continue training from existing model (default: False)', type=bool, default=False)
    parser.add_argument('-m', '--max_length', help='maximum sentence length (default: 500)', type=int, default=500)
    parser.add_argument('-d', '--dim', help='RNN state dimension (default: 1024)', type=int, default=1024)
    parser.add_argument('-w', '--n_words', help='vocabulary size (default: number of words in dictionary file)', type=int)
    parser.add_argument('-o', '--optimizer', help='optimizer (default: adam)', choices=['adam', 'rmsprop', 'adadelta', 'sgd'], default='adam')
    parser.add_argument('-l', '--learning_rate', help='learning rate (default: 0.0001)', type=float, default=0.0001)
    parser.add_argument('--clip_c', help='gradient norm clipping threshold, disabled if set to 0 (default: 1.0)', type=float, default=1.0)
    parser.add_argument('--decay_c', help='L2-regularization coefficient (default: 0.0)', type=float, default=0.0)
    parser.add_argument('--rnn', help='RNN type (default: gru)', choices=['gru'], default='gru')
    parser.add_argument('--rnn_rank', help='maximum rank of RNN recurrent matrices, integer or "full" (default: full)', default='full')
    parser.add_argument('--rnn_plus_diagonal', help='use low-rank plus diagonal parametrization, has effect only if rank is not "full" (default: True)', type=bool, default=True)
    parser.add_argument('--rnn_share_proj_matrix', help='share recurrent projection matrices between gates in low-rank and low-rank plus diagonal parametrizations (default: False)', type=bool, default=False)
    parser.add_argument('--dropout_retain_probability_word', help="retain probability for word embedding dropout (default: 0.9)", type=float, default=0.9)
    parser.add_argument('--dropout_retain_probability_rec', help="retain probability for recurrent matrices dropout (default: 0.5)", type=float, default=0.5)
    parser.add_argument('--dropout_retain_probability_readout', help="retain probability for readout layer dropout (default: 0.5)", type=float, default=0.5)
    parser.add_argument('--batch_size', help='training mini-batch size (default: 32)', type=int, default=32)
    parser.add_argument('--eval_batch_size', help='evaluation mini-batch size (default: 128)', type=int, default=128)
    parser.add_argument('--valid_freq', help='validation frequency in mini-batches (default: 5000)', type=int, default=5000)
    parser.add_argument('--display_freq', help='progress display frequency in mini-batches (default: 10)', type=int, default=10)
    parser.add_argument('--save_freq', help='best model save frequency in mini-batches (default: 1000)', type=int, default=1000)
    parser.add_argument('--patience', help='early stopping patience (default: 10)', type=int, default=10)

    args = parser.parse_args()
   
    train(
        saveto=args.model,
        reload_=args.reload,
        dim=args.dim,
        n_words=args.n_words,
        clip_c=args.clip_c,
        decay_c=args.decay_c,
        decoder=args.rnn,
        decoder_rank=('full' if args.rnn_rank == 'full' else int(args.rnn_rank)),
        decoder_plus_diagonal = args.rnn_plus_diagonal,
        decoder_share_proj_matrix = args.rnn_share_proj_matrix,
        retain_probability_word = args.dropout_retain_probability_word,
        retain_probability_rec = args.dropout_retain_probability_rec,
        retain_probability_readout = args.dropout_retain_probability_readout,
        lrate=args.learning_rate,
        optimizer=args.optimizer,
        maxlen=args.max_length,
        batch_size=args.batch_size,
        valid_batch_size=args.eval_batch_size,
        validFreq=args.valid_freq,
        dispFreq=args.display_freq,
        saveFreq=args.save_freq,
        patience=args.patience,
        dataset=args.train_dataset,
        valid_dataset=args.valid_dataset,
        test_dataset=args.test_dataset,
        dictionary=args.dictionary)

if __name__ == '__main__':
    main(0)

