from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import numpy as np
import tensorflow as tf
import heapq
import h5py

import lib
import model_base

########################################################################################
class NormalModel(model_base.Model):

    def __init__(self, dataset, init_method, min_init_weight, max_init_weight, embed_size, rnn_size, post_image_size, post_image_activation, rnn_type, learnable_init_state, multimodal_method, optimizer, learning_rate, normalize_image, weights_reg_weight, image_dropout_prob, post_image_dropout_prob, embedding_dropout_prob, rnn_dropout_prob, max_epochs, val_minibatch_size, train_minibatch_size):
        '''
        init_method: uniform, normal, xavier_uniform, xavier_normal
        post_image_activation: none, relu, tanh, swish
        rnn_type: srnn, gru, lstm
        multimodal_method: init, pre, par, merge
        optimizer: adam, rmsprop, adagrad
        '''
        super(NormalModel, self).__init__(dataset, val_minibatch_size)
        
        if init_method not in 'uniform, normal, xavier_uniform, xavier_normal'.split(', '):
            raise ValueError('Invalid init_method ({})'.format(init_method))
        if post_image_activation not in 'none, relu, tanh, swish'.split(', '):
            raise ValueError('Invalid post_image_activation ({})'.format(post_image_activation))
        if rnn_type not in 'srnn, gru, lstm'.split(', '):
            raise ValueError('Invalid rnn_type ({})'.format(rnn_type))
        if multimodal_method not in 'init, pre, par, merge'.split(', '):
            raise ValueError('Invalid multimodal_method ({})'.format(multimodal_method))
        if optimizer not in 'adam, rmsprop, adagrad'.split(', '):
            raise ValueError('Invalid optimizer ({})'.format(optimizer))
        if multimodal_method == 'init' and rnn_size != post_image_size:
            raise ValueError('Init multimodal method requires that rnn size and post image size be equal ({} != {})'.format(rnn_size, post_image_size))
        if multimodal_method == 'pre' and embed_size != post_image_size:
            raise ValueError('Pre multimodal method requires that embed size and post image size be equal ({} != {})'.format(embed_size, post_image_size))
        
        self.init_method             = init_method
        self.min_init_weight         = min_init_weight
        self.max_init_weight         = max_init_weight
        self.embed_size              = embed_size
        self.rnn_size                = rnn_size
        self.image_size              = 4096
        self.post_image_size         = post_image_size
        self.post_image_activation   = post_image_activation
        self.rnn_type                = rnn_type
        self.learnable_init_state    = learnable_init_state
        self.multimodal_method       = multimodal_method
        self.optimizer               = optimizer
        self.learning_rate           = learning_rate
        self.normalize_image         = normalize_image
        self.weights_reg_weight      = weights_reg_weight
        self.image_dropout_prob      = image_dropout_prob
        self.post_image_dropout_prob = post_image_dropout_prob
        self.embedding_dropout_prob  = embedding_dropout_prob
        self.rnn_dropout_prob        = rnn_dropout_prob
        self.max_epochs              = max_epochs
        self.train_minibatch_size    = train_minibatch_size
        
        self.prefixes      = None
        self.prefixes_lens = None
        self.images        = None
        self.temperature   = None
        self.dropout       = None
        self.targets       = None
        self.predictions   = None
        self.loss          = None
        self.session       = None
        self.train_step    = None
        self.initializer   = None
        self.param_setters = dict()
        self.num_params    = None

    ############################################

    def compile_model(self):
        if self.init_method in [ 'xavier_uniform', 'xavier_normal' ]:
            xavier = tf.contrib.layers.xavier_initializer(uniform=(self.init_method == 'xavier_uniform'), dtype=tf.float32)
        def init(shape, dtype=None, partition_info=None):
            if len(shape) == 1:
                return tf.zeros(shape, dtype=dtype)
            else:
                if self.init_method == 'uniform':
                    return tf.random_uniform(shape, self.min_init_weight, self.max_init_weight, dtype=dtype)
                elif self.init_method == 'normal':
                    return tf.clip_by_value(tf.random_normal(shape, dtype=dtype), self.min_init_weight, self.max_init_weight)
                elif self.init_method in [ 'xavier_uniform', 'xavier_normal' ]:
                    return tf.clip_by_value(xavier(shape, dtype, partition_info), self.min_init_weight, self.max_init_weight)
                    
        with tf.Graph().as_default():
            self.prefixes      = prefixes      = tf.placeholder(tf.int32,   [ None, None ],            'prefixes')
            self.prefixes_lens = prefixes_lens = tf.placeholder(tf.int32,   [ None ],                  'prefixes_lens')
            self.images        = images        = tf.placeholder(tf.float32, [ None, self.image_size ], 'images')
            self.dropout       = dropout       = tf.placeholder(tf.bool,    [],                        'dropout')
            self.temperature   = temperature   = tf.placeholder(tf.float32, [],                        'temperature')
            self.targets       = targets       = tf.placeholder(tf.int32,   [ None, None ],            'targets')

            batch_size = tf.shape(prefixes)[0]
            num_steps  = tf.shape(prefixes)[1]
            token_mask = tf.cast(tf.sequence_mask(prefixes_lens, num_steps), tf.float32)
            
            image_dropout_keep_prob      = tf.cond(dropout, lambda:tf.constant(1.0-self.image_dropout_prob), lambda:tf.constant(1.0))
            post_image_dropout_keep_prob = tf.cond(dropout, lambda:tf.constant(1.0-self.post_image_dropout_prob), lambda:tf.constant(1.0))
            embedding_dropout_keep_prob  = tf.cond(dropout, lambda:tf.constant(1.0-self.embedding_dropout_prob), lambda:tf.constant(1.0))
            rnn_dropout_keep_prob        = tf.cond(dropout, lambda:tf.constant(1.0-self.rnn_dropout_prob), lambda:tf.constant(1.0))
            
            with tf.variable_scope('nn', initializer=init):
                with tf.variable_scope('image'):
                    W = tf.get_variable('W', [ self.image_size, self.post_image_size ], tf.float32)
                    b = tf.get_variable('b', [ self.post_image_size ], tf.float32)
                    if self.normalize_image:
                        images = images/tf.reshape(tf.norm(images, axis=1), [ -1, 1 ]) #reshape is to divide the images row-wise instead of column-wise
                    images = tf.nn.dropout(images, image_dropout_keep_prob)
                    post_images = tf.matmul(images, W) + b
                    if self.post_image_activation == 'relu':
                        post_images = tf.nn.relu(post_images)
                    elif self.post_image_activation == 'tanh':
                        post_images = tf.nn.tanh(post_images)
                    elif self.post_image_activation == 'swish':
                        post_images = post_images*tf.nn.sigmoid(post_images)
                    if self.multimodal_method != 'init':
                        post_images = tf.expand_dims(post_images, 1)
                        if self.multimodal_method != 'pre':
                            post_images = tf.tile(post_images, [ 1, num_steps, 1 ])
                    post_images = tf.nn.dropout(post_images, post_image_dropout_keep_prob)

                with tf.variable_scope('prefix'):
                    with tf.variable_scope('embedding'):
                        embedding_matrix = tf.get_variable('embedding_matrix', [ self.dataset.vocab_size, self.embed_size ], tf.float32)
                        embedded_seq = tf.nn.embedding_lookup(embedding_matrix, prefixes)
                        embedded_seq = tf.nn.dropout(embedded_seq, embedding_dropout_keep_prob)
                        if self.multimodal_method == 'pre':
                            embedded_seq = tf.concat([ post_images, embedded_seq ], axis=1)
                        elif self.multimodal_method == 'par':
                            embedded_seq = tf.concat([ post_images, embedded_seq ], axis=2)

                    with tf.variable_scope('rnn'):
                        if self.rnn_type == 'srnn':
                            cell = tf.contrib.rnn.BasicRNNCell(self.rnn_size)
                        elif self.rnn_type == 'gru':
                            cell = tf.contrib.rnn.GRUCell(self.rnn_size)
                        elif self.rnn_type == 'lstm':
                            cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
                            
                        if self.multimodal_method == 'init':
                            h = post_images
                        else:
                            if self.learnable_init_state:
                                h = tf.get_variable('init_h', [ 1, self.rnn_size ], tf.float32)
                            else:
                                h = tf.zeros([ 1, self.rnn_size ], tf.float32)
                            h = tf.tile(h, [ batch_size, 1 ])
                        if self.rnn_type == 'lstm':
                            if self.learnable_init_state:
                                c = tf.get_variable('init_c', [ 1, self.rnn_size ], tf.float32)
                            else:
                                c = tf.zeros([ 1, self.rnn_size ], tf.float32)
                            c = tf.tile(c, [ batch_size, 1 ])
                            init_state = tf.contrib.rnn.LSTMStateTuple(h=h, c=c)
                        else:
                            init_state = h
                            
                    seq_len = prefixes_lens
                    if self.multimodal_method == 'pre':
                        seq_len = seq_len + 1
                    prefix_vectors = tf.nn.dynamic_rnn(cell, embedded_seq, sequence_length=seq_len, initial_state=init_state)[0] #Add 1 to prefixes_lens if using pre-inject since image is included as a token
                    prefix_vectors = tf.nn.dropout(prefix_vectors, rnn_dropout_keep_prob)
                    
                    if self.multimodal_method == 'merge':
                        prefix_vectors = tf.concat([ post_images, prefix_vectors ], axis=2)
                    elif self.multimodal_method == 'pre':
                        prefix_vectors = prefix_vectors[:,1:,:] #drop the prefix vector resulting from the image
                    
                    if self.multimodal_method == 'merge':
                        prefix_vector_size = self.rnn_size + self.post_image_size
                    else:
                        prefix_vector_size = self.rnn_size
                    prefix_vectors_2d = tf.reshape(prefix_vectors, [ batch_size*num_steps, prefix_vector_size ])
                    self.multimodal_vectors = prefix_vectors
                    
                with tf.variable_scope('out'):
                    W = tf.get_variable('W', [ prefix_vector_size, self.dataset.vocab_size ], tf.float32)
                    b = tf.get_variable('b', [ self.dataset.vocab_size ],                       tf.float32)
                    logits = tf.matmul(prefix_vectors_2d, W) + b
                    logits = tf.reshape(logits, [ batch_size, num_steps, self.dataset.vocab_size ])
                    
                    self.predictions = tf.nn.softmax(logits/temperature)
                
                with tf.variable_scope('loss'):
                    weights_reg = tf.nn.l2_loss(tf.concat([ tf.reshape(v, [-1]) for v in tf.trainable_variables() if len(v.shape) == 2 ], axis=0))
                    cross_ent = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits) * token_mask)
                    self.loss = cross_ent + self.weights_reg_weight*weights_reg

            if self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            elif self.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            self.train_step = optimizer.minimize(self.loss)
            
            self.initializer = tf.global_variables_initializer()
            
            self.num_params = 0
            for v in tf.trainable_variables():
                p = tf.placeholder(v.dtype, v.shape, v.name.split(':')[0]+'_setter')
                self.param_setters[v.name] = (tf.assign(v, p), p)
                
                self.num_params += np.prod(v.get_shape()).value
            
            self.session = tf.Session()
            tf.get_default_graph().finalize()

    ############################################

    def init_params(self):
        self.session.run(self.initializer)
    
    ############################################
    
    def save_params(self, param_save_dir):
        with self.session.graph.as_default():
            with h5py.File(param_save_dir+'/model.hdf5', 'w') as f:
                f.create_dataset('tf_version', data=np.array(tf.VERSION, np.string_))
                for (v, t) in zip(tf.trainable_variables(), self.session.run(tf.trainable_variables())):
                    f.create_dataset(v.name, data=t)
    
    ############################################

    def load_params(self, param_save_dir):
        with self.session.graph.as_default():
            self.init_params()
            with h5py.File(param_save_dir+'/model.hdf5', 'r') as f:
                loaded_tf_version = str(np.array(f['tf_version']))
                weights_biases_version = { '1.0.', '1.1.' }
                ts = dict()
                for v in tf.trainable_variables():
                    '''
                    Tensorflow v1.0-1.1 used 'weights' and 'biases' as variable names in RNNs whilst later versions use 'kernel' and 'bias' instead.
                    This is to avoid missing variable errors when loading parameters that were saved in a different version.
                    '''
                    name_parts = v.name.split('/')
                    if name_parts[2] == 'rnn':
                        if loaded_tf_version[:4] in weights_biases_version and tf.VERSION[:4] not in weights_biases_version:
                            if name_parts[4].startswith('weights'):
                                name_parts[4] = name_parts[4].replace('weights', 'kernel')
                            elif name_parts[4].startswith('biases'):
                                name_parts[4] = name_parts[4].replace('biases', 'bias')
                        elif loaded_tf_version[:4] not in weights_biases_version and tf.VERSION[:4] in weights_biases_version:
                            if name_parts[4].startswith('kernel'):
                                name_parts[4] = name_parts[4].replace('kernel', 'weights')
                            elif name_parts[4].startswith('bias'):
                                name_parts[4] = name_parts[4].replace('bias', 'biases')
                    ts['/'.join(name_parts)] = np.array(f[v.name])
                            
                self.session.run([ self.param_setters[v.name][0] for v in tf.trainable_variables() ], { self.param_setters[v.name][1]: ts[v.name] for v in tf.trainable_variables() })
    
    ############################################

    def fit(self, param_save_dir, listener=model_base.FitListener()):
        listener.fit_started(self)

        last_validation_loss = np.inf
        for epoch in range(0, self.max_epochs+1):
            listener.epoch_started(self, epoch)

            #Training
            if epoch > 0:
                listener.training_started(self)
                    
                trainingset_indexes = np.arange(self.dataset.training_datasource.size)
                np.random.shuffle(trainingset_indexes)
                num_minibatches = int(np.ceil(self.dataset.training_datasource.size/self.train_minibatch_size))
                for i in range(num_minibatches):
                    listener.minibatch_started(self, i+1, num_minibatches)
                        
                    minibatch_indexes = trainingset_indexes[i*self.train_minibatch_size:(i+1)*self.train_minibatch_size]
                    feed_dict = {
                            self.dropout:       True,
                            self.temperature:   1.0,
                            self.prefixes:      self.dataset.training_proccaps.prefixes_indexes[minibatch_indexes],
                            self.prefixes_lens: self.dataset.training_proccaps.prefixes_lens[minibatch_indexes],
                            self.images:        self.dataset.training_images[minibatch_indexes],
                            self.targets:       self.dataset.training_proccaps.targets_indexes[minibatch_indexes],
                        }
                    self.session.run(self.train_step, feed_dict=feed_dict)
                    
                    listener.minibatch_ended(self, i+1, num_minibatches)
                        
                listener.training_ended(self)
            
            #Validation
            listener.validation_started(self)
                    
            validation_loss = self.image_caption_loggeomean_perplexities(self.dataset.validation_datasource.images, self.dataset.validation_datasource.first_captions)[0]
            
            listener.validation_ended(self, validation_loss)

            if validation_loss > last_validation_loss:
                listener.epoch_ended(self, epoch)
                break
            else:
                last_validation_loss = validation_loss
                self.save_params(param_save_dir)
                listener.epoch_ended(self, epoch)
        
        self.load_params(param_save_dir)
        
        listener.fit_ended(self)

        return last_validation_loss

    ############################################
    
    def get_num_params(self):
        return self.num_params
        
    ############################################
    
    def raw_run(self, node, images, prefixes, prefixes_lens, temperature):
        return self.session.run(
                node,
                feed_dict={
                        self.dropout:       False,
                        self.temperature:   temperature,
                        self.prefixes:      prefixes,
                        self.prefixes_lens: prefixes_lens,
                        self.images:        images,
                    }
            )
    
    ############################################
    
    def get_raw_probs(self, images, prefixes, prefixes_lens, temperature):
        return self.raw_run(self.predictions, images, prefixes, prefixes_lens, temperature)
    
    ############################################

    def close(self):
        #if self.session is not None:
        #   self.session.close()
        tf.Session.reset('')