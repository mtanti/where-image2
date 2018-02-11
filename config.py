debug = True

def raw_data_dir(dataset_name):
    return '.../datasets/{}'.format(dataset_name) #Karpathy raw data expected (http://cs.stanford.edu/people/karpathy/deepimagesent/)

mscoco_dir = '.../tools/coco-caption-master' #MSCOCO evaluation toolkit expected (https://github.com/tylin/coco-caption)

base_dir          = 'results'     if not debug else 'results_test'
base_dir_hyperpar = 'hyperparams' if not debug else 'hyperparams_test'

min_token_freq     = 5
max_epochs         = 20 if not debug else 2
num_runs           = 3  if not debug else 2
val_minibatch_size = 400

beamsearch_lower_bound_len = 5
beamsearch_upper_bound_len = 50
beamsearch_temperature     = 1.0

hyperpar_max_epochs         = 10  if not debug else 2
hyperpar_beam_width         = 2   if not debug else 1
hyperpar_max_evals          = 100 if not debug else 2
hyperpar_finetune_iters     = 5   if not debug else 1
hyperpar_num_candidate_hyps = 10
hyperpar_num_best_hyps      = 3   if not debug else 2
hyperpar_max_beam_width     = 5   if not debug else 2

hyperparams = {

    'init': dict(
        init_method             = 'xavier_normal',
        min_init_weight         = -0.01,
        max_init_weight         = 0.01,
        embed_size              = 512,
        rnn_size                = 512,
        post_image_size         = 512,
        post_image_activation   = 'none',
        rnn_type                = 'gru',
        learnable_init_state    = True,
        optimizer               = 'adam',
        learning_rate           = 0.001,
        normalize_image         = True,
        weights_reg_weight      = 0.0,
        image_dropout_prob      = 0.0,
        post_image_dropout_prob = 0.0,
        embedding_dropout_prob  = 0.5,
        rnn_dropout_prob        = 0.5,
        train_minibatch_size    = 128,
        max_epochs              = 100,
        beam_width              = 3,
    ),
    
    'pre': dict(
        init_method             = 'normal',
        min_init_weight         = -0.1,
        max_init_weight         = 0.1,
        embed_size              = 512,
        rnn_size                = 512,
        post_image_size         = 512,
        post_image_activation   = 'none',
        rnn_type                = 'gru',
        learnable_init_state    = False,
        optimizer               = 'adam',
        learning_rate           = 0.001,
        normalize_image         = True,
        weights_reg_weight      = 0.0,
        image_dropout_prob      = 0.0,
        post_image_dropout_prob = 0.0,
        embedding_dropout_prob  = 0.5,
        rnn_dropout_prob        = 0.5,
        train_minibatch_size    = 32,
        max_epochs              = 100,
        beam_width              = 3,
    ),
    
    'par': dict(
        init_method             = 'normal',
        min_init_weight         = -0.1,
        max_init_weight         = 0.1,
        embed_size              = 256,
        rnn_size                = 256,
        post_image_size         = 256,
        post_image_activation   = 'none',
        rnn_type                = 'gru',
        learnable_init_state    = True,
        optimizer               = 'adam',
        learning_rate           = 0.001,
        normalize_image         = True,
        weights_reg_weight      = 1e-08,
        image_dropout_prob      = 0.5,
        post_image_dropout_prob = 0.0,
        embedding_dropout_prob  = 0.5,
        rnn_dropout_prob        = 0.5,
        train_minibatch_size    = 64,
        max_epochs              = 100,
        beam_width              = 5,
    ),
    
    'merge': dict(
        init_method             = 'normal',
        min_init_weight         = -0.1,
        max_init_weight         = 0.1,
        embed_size              = 128,
        rnn_size                = 128,
        post_image_size         = 128,
        post_image_activation   = 'none',
        rnn_type                = 'gru',
        learnable_init_state    = True,
        optimizer               = 'adam',
        learning_rate           = 0.001,
        normalize_image         = True,
        weights_reg_weight      = 0.0,
        image_dropout_prob      = 0.0,
        post_image_dropout_prob = 0.0,
        embedding_dropout_prob  = 0.0,
        rnn_dropout_prob        = 0.5,
        train_minibatch_size    = 128,
        max_epochs              = 100,
        beam_width              = 3,
    ),
    
}