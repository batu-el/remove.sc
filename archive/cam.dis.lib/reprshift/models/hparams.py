TEXT_DATASETS = ["CivilCommentsFine", "MultiNLI", "CivilComments"]

HPARAM = {'batch_size': 32,
          'last_layer_dropout': 0.5,
          'optimizer': 'adamw',
          'weight_decay':1e-4,
          'lr': 1e-5,
          'group_balanced': False,
          'num_training_steps': 30001,
          'num_warmup_steps': 0,
          }

MODEL_HPARAM = { 'ERM'          : {},
                 'IRM'          : {'irm_lambda': 1e2, 'irm_penalty_anneal_iters': 500},
                 'GroupDRO'     : {'groupdro_eta':  1e-2},
                 'CVaRDRO'      : {'joint_dro_alpha': 0.1},
                 'Mixup'        : {'mixup_alpha': 0.2},
                 'JTT'          : {"first_stage_step_frac": 0.5 , "jtt_lambda": 10},
                 'LfF'          : {'batch_size':16, 'LfF_q': 0.7},
                 'LISA'         : {'LISA_alpha': 2., 'LISA_p_sel': 0.5, 'LISA_mixup_method': 'mixup'},
                 'DFR'          : {'stage1_model': 'model.pkl'},
                 'MDD'          : {'mmd_gamma': 1.},
                 'CORAL'        : {'CORAL': 1.},
                 'ReSample'     : {'group_balanced' : True},
                 'ReWeight'     : {},
                 'SqrtReWeight' : {},
                 'Focal'        : {'gamma': 1},
                 'CBLoss'       : {'beta': 1 - 1e-4},
                 'LDAM'         : {'max_m': 0.5, 'scale': 30},
                 'BSoftmax'     : {},
                 'CRT'          : {'group_balanced' : True, 'stage1_model': 'model.pkl'},
                 'CNC'          : {},
                 }

HPARAM_SEARCH = {'batch_size': [32],
                'last_layer_dropout': [0., 0.1, 0.5],
                  'optimizer': ['adamw'],
                 'weight_decay':[1e-5, 1e-4, 1e-3],
                 'lr': [1e-6, 1e-5, 1e-4],
                 }

def hparams_f(model_name):
    hparams = HPARAM
    hparams_model =  MODEL_HPARAM[model_name]

    # overwriting the initial hyperparameters when necessary
    for key in hparams_model:
        hparams[key] = hparams_model[key]
    
    return hparams