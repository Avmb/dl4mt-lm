'''
Build a simple neural language model using GRU units, without embedding layers or non-linear readout
'''
import theano
import theano.tensor as tensor
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle as pkl
import ipdb
import numpy
import copy

import os
import warnings
import sys
import time

from collections import OrderedDict

from data_iterator import TextIterator

profile = False


# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout
def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(
        use_noise,
        state_before * trng.binomial(state_before.shape, p=0.5, n=1,
                                     dtype=state_before.dtype),
        state_before * 0.5)
    return proj

# dropout that will be re-used at different time steps
def shared_dropout_layer(shape, use_noise, trng, value):
    proj = tensor.switch(
        use_noise,
        trng.binomial(shape, p=value, n=1,
                                     dtype='float32') / theano.shared(numpy.float32(value)),
        1.0)
    return proj

# make prefix-appended name
def pp(prefix, name):
    return '%s_%s' % (prefix, name)


# initialize Theano shared variables according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]

    return params


# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          }


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# orthogonal initialization for weights
# see Saxe et al. ICLR'14
def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype('float32')


# weight initializer, normal by default
def norm_weight(nin, nout=None, scale=0.01, ortho=True):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype('float32')


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = tensor.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = tensor.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


# batch preparation, returns padded batch and mask
def prepare_data(seqs_x, maxlen=None, n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]

    # filter according to mexlen
    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    for idx, s_x in enumerate(seqs_x):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.

    return x, x_mask


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None,
                       ortho=True):
    if nin is None:
        nin = options['dim_proj']
    if nout is None:
        nout = options['dim_proj']
    params[pp(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    params[pp(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams, state_below, options, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', **kwargs):
    return eval(activ)(
        tensor.dot(state_below, tparams[pp(prefix, 'W')]) +
        tparams[pp(prefix, 'b')])


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None, rank='full', share_proj_matrix=False, plus_diagonal=True):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[pp(prefix, 'W')] = W
    params[pp(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # embedding to hidden state proposal weights, biases
    Wx = norm_weight(nin, dim)
    params[pp(prefix, 'Wx')] = Wx
    params[pp(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    if rank == 'full':
        # recurrent transformation weights for gates
        U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
        params[pp(prefix, 'U')] = U

        # recurrent transformation weights for hidden state proposal
        Ux = ortho_weight(dim)
        params[pp(prefix, 'Ux')] = Ux
    else:
        if share_proj_matrix:
            U_proj = norm_weight(dim, rank)
            params[pp(prefix, 'U_proj')] = U_proj
        else:
            U_proj_u = norm_weight(dim, rank)
            U_proj_r = norm_weight(dim, rank)
            U_proj_x = norm_weight(dim, rank)
            params[pp(prefix, 'U_proj_u')] = U_proj_u
            params[pp(prefix, 'U_proj_r')] = U_proj_r
            params[pp(prefix, 'U_proj_x')] = U_proj_x
        U_expand_u = norm_weight(rank, dim)
        U_expand_r = norm_weight(rank, dim)
        U_expand_x = norm_weight(rank, dim)
        params[pp(prefix, 'U_expand_u')] = U_expand_u
        params[pp(prefix, 'U_expand_r')] = U_expand_r
        params[pp(prefix, 'U_expand_x')] = U_expand_x
        if plus_diagonal:
            U_diag_u = numpy.random.uniform(size=dim, low=-0.01, high=0.01).astype('float32')
            U_diag_r = numpy.random.uniform(size=dim, low=-0.01, high=0.01).astype('float32')
            U_diag_x = numpy.random.uniform(size=dim, low=-0.01, high=0.01).astype('float32')
            params[pp(prefix, 'U_diag_u')] = U_diag_u
            params[pp(prefix, 'U_diag_r')] = U_diag_r
            params[pp(prefix, 'U_diag_x')] = U_diag_x

    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None,
              profile=False,
              integer_input=False,
              word_dropout=None,
              rec_dropout=None,
              rank='full',
              share_proj_matrix=False,
              plus_diagonal=True,
              **kwargs):
    nsteps = state_below.shape[0]
    minibatched_mode_ndim = 2 if integer_input else 3
    if state_below.ndim == minibatched_mode_ndim:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[pp(prefix, 'Wx')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    if integer_input:
        state_below_flat = state_below.flatten()
        # input to the gates, concatenated
        state_below_ = tparams[pp(prefix, 'W')][state_below_flat] + tparams[pp(prefix, 'b')]
        pre_shape_ = (nsteps, n_samples, 2*dim)
        state_below_ = state_below_.reshape(pre_shape_)
        # input to compute the hidden state proposal
        state_belowx = tparams[pp(prefix, 'Wx')][state_below_flat] + tparams[pp(prefix, 'bx')]
        pre_shapex = (nsteps, n_samples, dim)
        state_belowx = state_belowx.reshape(pre_shapex)
        if word_dropout != None:
            state_below_ *= word_dropout
            state_belowx *= word_dropout
    else:
        # state_below is the input word embeddings
        # input to the gates, concatenated
        state_below_ = tensor.dot(state_below, tparams[pp(prefix, 'W')]) + \
            tparams[pp(prefix, 'b')]
        # input to compute the hidden state proposal
        state_belowx = tensor.dot(state_below, tparams[pp(prefix, 'Wx')]) + \
            tparams[pp(prefix, 'bx')]
            
    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_):
        h_.name='h_'
        if rank == 'full':
            preact = tensor.dot(h_ * rec_dropout[0], tparams[pp(prefix, 'U')])
            preact.name='preact'
            preact += x_

            # reset and update gates
            r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
            u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

            # compute the hidden state proposal
            preactx = tensor.dot(h_ * rec_dropout[1], tparams[pp(prefix, 'Ux')])
            preactx = preactx * r
            preactx = preactx + xx_
        else:
            if share_proj_matrix:
                h_dr = h_ * rec_dropout[0] if rec_dropout != None else h_
                hu_dr, hr_dr, hx_dr = h_dr, h_dr, h_dr
                proj = tensor.dot(h_dr, tparams[pp(prefix, 'U_proj')])
                proj_u, proj_r, proj_x = proj, proj, proj
            else:
                hu_dr = h_ * rec_dropout[0] if rec_dropout != None else h_
                hr_dr = h_ * rec_dropout[1] if rec_dropout != None else h_
                hx_dr = h_ * rec_dropout[2] if rec_dropout != None else h_
                proj_u = tensor.dot(hu_dr, tparams[pp(prefix, 'U_proj_u')])
                proj_r = tensor.dot(hr_dr, tparams[pp(prefix, 'U_proj_r')])
                proj_x = tensor.dot(hx_dr, tparams[pp(prefix, 'U_proj_x')])
            preact_u = tensor.dot(proj_u, tparams[pp(prefix, 'U_expand_u')]) + _slice(x_, 0, dim)
            preact_r = tensor.dot(proj_r, tparams[pp(prefix, 'U_expand_r')]) + _slice(x_, 1, dim)
            if plus_diagonal:
                 preact_u += hu_dr * tparams[pp(prefix, 'U_diag_u')]
                 preact_r += hr_dr * tparams[pp(prefix, 'U_diag_r')]
            u = tensor.nnet.sigmoid(preact_u)
            r = tensor.nnet.sigmoid(preact_r)
            pre_preactx = tensor.dot(proj_x, tparams[pp(prefix, 'U_expand_x')])
            if plus_diagonal:
                pre_preactx += hx_dr * tparams[pp(prefix, 'U_diag_x')]
            preactx = pre_preactx * r + xx_
            

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    init_states = [tensor.alloc(0., n_samples, dim)]
    _step = _step_slice

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                name=pp(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=profile,
                                strict=False)
    rval = [rval]
    return rval


# initialize all parameters
def init_params(options):
    params = OrderedDict()
    # embedding
    params = get_layer(options['decoder'])[0](options, params,
                                              prefix='decoder',
                                              nin=options['n_words'],
                                              dim=options['dim'],
                                              rank=options['decoder_rank'],
                                              share_proj_matrix=options['decoder_share_proj_matrix'],
                                              plus_diagonal=options['decoder_plus_diagonal'])
    # readout
    params = get_layer('ff')[0](options, params, prefix='ff_out',
                                nin=options['dim'], nout=options['n_words'],
                                ortho=False)
    return params


# build a training model
def build_model(tparams, options):
    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # dropout
    n_rec_dropout_masks = 1 if ((options['decoder_rank'] != 'full') and options['decoder_share_proj_matrix']) else 3
    word_dropout = shared_dropout_layer((n_timesteps, n_samples), use_noise, trng, options['retain_probability_word'])
    word_dropout = tensor.shape_padright(word_dropout)
    rec_dropout = shared_dropout_layer((n_rec_dropout_masks, n_samples, options['dim']), use_noise, trng, options['retain_probability_rec'])
    readout_dropout = shared_dropout_layer((n_samples, options['dim']), use_noise, trng, options['retain_probability_readout'])
    readout_dropout = tensor.shape_padleft(readout_dropout)

    # input

    x_shifted = tensor.zeros_like(x, dtype='int64')
    x_shifted = tensor.set_subtensor(x_shifted[1:], x[:-1])
    opt_ret['x_shifted'] = x_shifted

    # pass through gru layer, recurrence here
    proj = get_layer(options['decoder'])[1](tparams, x_shifted, options,
                                            prefix='decoder',
                                            mask=x_mask,
                                            integer_input=True,
                                            profile=profile,
                                            word_dropout=word_dropout,
                                            rec_dropout=rec_dropout,
                                            rank=options['decoder_rank'],
                                            share_proj_matrix=options['decoder_share_proj_matrix'],
                                            plus_diagonal=options['decoder_plus_diagonal'])
    proj_h = proj[0]
    opt_ret['proj_h'] = proj_h

    # compute word probabilities
    logit = get_layer('ff')[1](tparams, proj_h*readout_dropout, options,
                                    prefix='ff_out', activ='linear')
    logit_shp = logit.shape
    probs = tensor.nnet.softmax(
        logit.reshape([logit_shp[0]*logit_shp[1], logit_shp[2]]))

    # cost
    x_flat = x.flatten()
    x_flat_idx = tensor.arange(x_flat.shape[0]) * options['n_words'] + x_flat
    cost = -tensor.log(probs.flatten()[x_flat_idx])
    cost = cost.reshape([x.shape[0], x.shape[1]])
    opt_ret['cost_per_sample'] = cost
    #cost = (cost * x_mask).sum(0)
    #cost = (cost * x_mask).mean(0)   # Average cost over words (wrong)
    cost = (cost * x_mask).sum(0) / x_mask.sum(0) # Average cost over words

    return trng, use_noise, x, x_mask, opt_ret, cost

# calculate the log probablities on a given corpus using language model
def pred_probs(f_log_probs, prepare_data, options, iterator, verbose=True):
    probs = []

    n_done = 0

    for x in iterator:
        n_done += len(x)

        x, x_mask = prepare_data(x, n_words=options['n_words'])

        pprobs = f_log_probs(x, x_mask)
        for pp in pprobs:
            probs.append(pp)

        if numpy.isnan(numpy.mean(probs)):
            ipdb.set_trace()

        if verbose:
            print >> sys.stderr, '%d samples computed' % (n_done)

    return numpy.array(probs)


# optimizers
# name(hyperp, tparams, grads, inputs (list), cost) = f_grad_shared, f_update
def adam(lr, tparams, grads, inp, cost, beta1=0.9, beta2=0.999, e=1e-8):

    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    f_grad_shared = theano.function(inp, cost, updates=gsup, profile=profile)

    updates = []

    t_prev = theano.shared(numpy.float32(0.))
    t = t_prev + 1.
    lr_t = lr * tensor.sqrt(1. - beta2**t) / (1. - beta1**t)

    for p, g in zip(tparams.values(), gshared):
        m = theano.shared(p.get_value() * 0., p.name + '_mean')
        v = theano.shared(p.get_value() * 0., p.name + '_variance')
        m_t = beta1 * m + (1. - beta1) * g
        v_t = beta2 * v + (1. - beta2) * g**2
        step = lr_t * m_t / (tensor.sqrt(v_t) + e)
        p_t = p - step
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((t_prev, t))

    f_update = theano.function([lr], [], updates=updates,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up,
                                    profile=profile)

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rgup+rg2up,
                                    profile=profile)

    updir = [theano.shared(p.get_value() * numpy.float32(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(itemlist(tparams), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new+param_up,
                               on_unused_input='ignore', profile=profile)

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, x, mask, y, cost):

    # allocate gradients and set them all to zero
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]

    # create gradient copying list,
    # from grads (tensor variable) to gshared (shared variable)
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # compile theano function to compute cost and copy gradients
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    profile=profile)

    # define the update step rule
    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]

    # compile a function for update
    f_update = theano.function([lr], [], updates=pup, profile=profile)

    return f_grad_shared, f_update


def train(
          dim=1000,  # the number of GRU units
          decoder='gru',
          patience=10,  # early stopping patience
          max_epochs=5000,
          finish_after=10000000,  # finish after this many updates
          dispFreq=100,
          clip_c=1.0,
          decay_c=0.,  # L2 weight decay penalty
          lrate=0.01,
          nan_guard=0.01, # perform a L2 regularization update with this penalty when a NaN or Inf is detected in the gradient. Disabled if equal to 0.0
          n_words=None,  # vocabulary size
          maxlen=100,  # maximum length of the description
          optimizer='rmsprop',
          batch_size=16,
          valid_batch_size=16,
          saveto='model.npz',
          validFreq=1000,
          saveFreq=1000,  # save the parameters after every saveFreq updates
          #sampleFreq=100,  # generate some samples after every sampleFreq
          dataset='train.txt.gz',
          valid_dataset='valid.txt',
          test_dataset=None,
          dictionary='vocab.pkl',
          #use_dropout=False,
          reload_=False,
          decoder_rank='full', # 'full' or integer. If it is integer enables low-rank or low-rank plus diagonal parametrization (Miceli Barone 2016) arXiv:1603.03116
          decoder_share_proj_matrix=False, # in low-rank or low-rank plus diagonal mode, control whether the projection matrices are shared between the proposal, reset and update gates of the GRU
          decoder_plus_diagonal=True, # switch between low-rank and low-rank plus diagonal
          retain_probability_word=0.9,
          retain_probability_rec=0.5,
          retain_probability_readout=0.5,
          ):

    # Model options
    model_options = locals().copy()
    print >> sys.stderr, model_options

    # load dictionary
    with open(dictionary, 'rb') as f:
        worddicts = pkl.load(f)

    # invert dictionary
    worddicts_r = dict()
    for kk, vv in worddicts.iteritems():
        worddicts_r[vv] = kk

    # compute dictionary size if not supplied as a parameter
    if n_words is None:
	n_words = len(worddicts)
        model_options['n_words'] = n_words

    # reload options
    if reload_ and os.path.exists(saveto):
        with open('%s.pkl' % saveto, 'rb') as f:
            model_options = pkl.load(f)

    print >> sys.stderr, 'Loading data'
    train = TextIterator(dataset,
                         dictionary,
                         n_words_source=n_words,
                         batch_size=batch_size,
                         maxlen=maxlen)
    valid = TextIterator(valid_dataset,
                         dictionary,
                         n_words_source=n_words,
                         batch_size=valid_batch_size,
                         maxlen=maxlen)
    test = TextIterator(test_dataset,
                         dictionary,
                         n_words_source=n_words,
                         batch_size=valid_batch_size,
                         maxlen=maxlen) if test_dataset != None else None

    print >> sys.stderr, 'Building model'
    params = init_params(model_options)

    # reload parameters
    if reload_ and os.path.exists(saveto):
        params = load_params(saveto, params)

    # create shared variables for parameters
    tparams = init_tparams(params)

    # build the symbolic computational graph
    trng, use_noise, \
        x, x_mask, \
        opt_ret, \
        cost = \
        build_model(tparams, model_options)
    inps = [x, x_mask]

    #print >> sys.stderr, 'Buliding sampler'
    #f_next = build_sampler(tparams, model_options, trng)

    # before any regularizer
    print >> sys.stderr, 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=profile)
    print >> sys.stderr, 'Done'

    cost = cost.mean()

    # apply L2 regularization on weights
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer - compile the computational graph for cost
    print >> sys.stderr, 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=profile)
    print >> sys.stderr, 'Done'

    print >> sys.stderr, 'Computing gradient...',
    param_list = itemlist(tparams)
    grads = tensor.grad(cost, wrt=param_list)
    print >> sys.stderr, 'Done'

    # apply gradient clipping here
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        isnan = tensor.or_(tensor.isnan(g2), tensor.isinf(g2))
        new_grads = []
        for i, g in enumerate(grads):
            ng = ifelse(g2 > (clip_c**2), (g / tensor.sqrt(g2) * clip_c), g)
            if nan_guard > 0.0:
                altg = nan_guard * param_list[i]
                ng = ifelse(isnan, altg, ng)
            new_grads.append(ng)
        grads = new_grads

    # compile the optimizer, the actual computational graph is compiled here
    lr = tensor.scalar(name='lr')
    print >> sys.stderr, 'Building optimizers...',
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)
    print >> sys.stderr, 'Done'

    print >> sys.stderr, 'Optimization'

    history_errs = []
    # reload history
    if reload_ and os.path.exists(saveto):
        history_errs = list(numpy.load(saveto)['history_errs'])
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0])/batch_size
    if saveFreq == -1:
        saveFreq = len(train[0])/batch_size
    #if sampleFreq == -1:
    #    sampleFreq = len(train[0])/batch_size

    # Training loop
    uidx = 0
    estop = False
    bad_counter = 0
    best_valid_err = numpy.inf
    for eidx in xrange(max_epochs):
        n_samples = 0

        for x in train:
            n_samples += len(x)
            uidx += 1
            use_noise.set_value(1.)

            # pad batch and create mask
            x, x_mask = prepare_data(x, maxlen=maxlen, n_words=n_words)

            if x is None:
                print >> sys.stderr, 'Minibatch with zero sample under length ', maxlen
                uidx -= 1
                continue

            ud_start = time.time()

            # compute cost, grads and copy grads to shared variables
            cost = f_grad_shared(x, x_mask)

            # do the update on parameters
            f_update(lrate)

            ud = time.time() - ud_start

            # check for bad numbers
            if numpy.isnan(cost) or numpy.isinf(cost):
                print >> sys.stderr, 'NaN detected'

            # verbose
            if numpy.mod(uidx, dispFreq) == 0:
                print >> sys.stderr, 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud

            # save the best model so far
            if numpy.mod(uidx, saveFreq) == 0:
                print >> sys.stderr, 'Saving...',

                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_errs, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                print >> sys.stderr, 'Done'

            # generate some samples with the model and display them
            #if numpy.mod(uidx, sampleFreq) == 0:
            #    # FIXME: random selection?
            #    for jj in xrange(5):
            #        sample, score = gen_sample(tparams, f_next,
            #                                   model_options, trng=trng,
            #                                   maxlen=30, argmax=False)
            #        print >> sys.stderr, 'Sample ', jj, ': ',
            #        ss = sample
            #        for vv in ss:
            #            if vv == 0:
            #                break
            #            if vv in worddicts_r:
            #                print >> sys.stderr, worddicts_r[vv],
            #            else:
            #                print >> sys.stderr, 'UNK',
            #        print

            # validate model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                ud_start = time.time()
                valid_errs = pred_probs(f_log_probs, prepare_data,
                                        model_options, valid)
                valid_err = valid_errs.mean()
                history_errs.append(valid_err)
                ud = time.time() - ud_start

                if uidx == 0 or valid_err <= best_valid_err:
                    best_p = unzip(tparams)
                    bad_counter = 0
                    best_valid_err = valid_err
                else:
                    bad_counter += 1
                    if bad_counter > patience:
                        print >> sys.stderr, 'Early Stop!'
                        estop = True
                        break

                if numpy.isnan(valid_err):
                    ipdb.set_trace()

                print >> sys.stderr, 'Validation loss:', valid_err, 'perplexity:', numpy.exp(valid_err), 'UD:', ud,
                if bad_counter == 0:
                    print >> sys.stderr, '*** NEW BEST ***'
                else:
                    print >> sys.stderr, ''

            # finish after this many updates
            if uidx >= finish_after:
                print >> sys.stderr, 'Finishing after %d iterations!' % uidx
                estop = True
                break

        print >> sys.stderr, 'Seen %d samples' % n_samples

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    ud_start = time.time()
    valid_err = pred_probs(f_log_probs, prepare_data,
                           model_options, valid).mean()
    ud = time.time() - ud_start

    print >> sys.stderr, 'Validation loss:', valid_err, 'perplexity:', numpy.exp(valid_err), 'UD:', ud

    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_errs,
                **params)

    if test != None:
        print >> sys.stderr, 'Testing...'
        use_noise.set_value(0.)
        ud_start = time.time()
        test_err = pred_probs(f_log_probs, prepare_data,
                           model_options, test).mean()
        ud = time.time() - ud_start

        print >> sys.stderr, 'Test loss:', test_err, 'perplexity:', numpy.exp(test_err), 'UD:', ud
        

    return valid_err


if __name__ == '__main__':
    pass
