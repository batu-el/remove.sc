def ModelDictMap():
    ModelDictMap = []

    EmbeddingsMap = [### Embeddings ###
                    ('network.0.featurizer.embeddings.word_embeddings.weight','embed.embed.W_E'),
                    ('network.0.featurizer.embeddings.position_embeddings.weight', 'embed.pos_embed.W_pos'),
                    ('network.0.featurizer.embeddings.token_type_embeddings.weight',  'embed.token_type_embed.W_token_type'),
                    ('network.0.featurizer.embeddings.LayerNorm.weight','embed.ln.w'),
                    ('network.0.featurizer.embeddings.LayerNorm.bias','embed.ln.b'),]
    LayersMap = lambda i :[ ### Layer i ###
                            ### Query ###
                            (f'network.0.featurizer.encoder.layer.{i}.attention.self.query.weight', f'blocks.{i}.attn.W_Q'),
                            (f'network.0.featurizer.encoder.layer.{i}.attention.self.query.bias', f'blocks.{i}.attn.b_Q'),
                            ### Key ###
                            (f'network.0.featurizer.encoder.layer.{i}.attention.self.key.weight', f'blocks.{i}.attn.W_K'),
                            (f'network.0.featurizer.encoder.layer.{i}.attention.self.key.bias', f'blocks.{i}.attn.b_K'),
                            ### Value ###
                            (f'network.0.featurizer.encoder.layer.{i}.attention.self.value.weight', f'blocks.{i}.attn.W_V'),
                            (f'network.0.featurizer.encoder.layer.{i}.attention.self.value.bias', f'blocks.{i}.attn.b_V'),
                            ### Out ###
                            (f'network.0.featurizer.encoder.layer.{i}.attention.output.dense.weight', f'blocks.{i}.attn.W_O'),
                            (f'network.0.featurizer.encoder.layer.{i}.attention.output.dense.bias', f'blocks.{i}.attn.b_O'),
                            ### LayerNorm ###
                            (f'network.0.featurizer.encoder.layer.{i}.attention.output.LayerNorm.weight', f'blocks.{i}.ln1.w'),
                            (f'network.0.featurizer.encoder.layer.{i}.attention.output.LayerNorm.bias',f'blocks.{i}.ln1.b'),
                            ### MLP in ###
                            (f'network.0.featurizer.encoder.layer.{i}.intermediate.dense.weight', f'blocks.{i}.mlp.W_in'),
                            (f'network.0.featurizer.encoder.layer.{i}.intermediate.dense.bias', f'blocks.{i}.mlp.b_in'),
                            ### MLP out ###
                            (f'network.0.featurizer.encoder.layer.{i}.output.dense.weight',f'blocks.{i}.mlp.W_out'),
                            (f'network.0.featurizer.encoder.layer.{i}.output.dense.bias',f'blocks.{i}.mlp.b_out'),
                            ### LayerNorm 2 ###
                            (f'network.0.featurizer.encoder.layer.{i}.output.LayerNorm.weight', f'blocks.{i}.ln2.w'),
                            (f'network.0.featurizer.encoder.layer.{i}.output.LayerNorm.bias', f'blocks.{i}.ln2.b'),]
    PoolerMap = [ ### Pooler ###
                ('network.0.featurizer.pooler.dense.weight','mlm_head.W'),
                ('network.0.featurizer.pooler.dense.bias','mlm_head.b'),
                # ( None,'mlm_head.ln.w'),
                # ( None,'mlm_head.ln.b'),
                ]

    UnembeddingsMap = [ ### Unembeddings ###
                        ('network.1.classifier.weight','unembed.W_U'),
                        ('network.1.classifier.bias','unembed.b_U'),]

    ### Model Dict Map ###
    ModelDictMap = []
    ModelDictMap += EmbeddingsMap
    for i in range(12):
        ModelDictMap += LayersMap(i)
    ModelDictMap += PoolerMap
    ModelDictMap += UnembeddingsMap

    return ModelDictMap

def ERM_to_HookedEncoder(sd_ERM, sd_HookedEncoder):
    mdm = ModelDictMap()
    out_sd_ERM = {}

    for param_name in mdm:
        out_sd_ERM[param_name[1]] = sd_ERM[param_name[0]]
    
    for param_name in list(sd_HookedEncoder.keys()):
        if param_name not in out_sd_ERM.keys():
            out_sd_ERM[param_name] = sd_HookedEncoder[param_name]
    
    for param_name in list(out_sd_ERM.keys()):
        current_shape = out_sd_ERM[param_name].shape
        target_shape = sd_HookedEncoder[param_name].shape
        if current_shape != target_shape:
            if 'attn.W_O' in param_name:
                out_sd_ERM[param_name] = out_sd_ERM[param_name].reshape(768, 12, 64).transpose(0,1).transpose(1,2)
            elif 'attn.W' in param_name:
                out_sd_ERM[param_name] = out_sd_ERM[param_name].reshape(12, 64, 768).transpose(1,2)
            elif 'attn.b' in param_name:
                out_sd_ERM[param_name] = out_sd_ERM[param_name].reshape(12, 64)
            elif 'mlp.W_in' in param_name:
                out_sd_ERM[param_name] = out_sd_ERM[param_name].T
            elif 'mlp.W_out' in param_name:
                out_sd_ERM[param_name] = out_sd_ERM[param_name].T
            elif 'unembed.W_U' in param_name:
                out_sd_ERM[param_name] = out_sd_ERM[param_name].T
            else:
                pass
    return out_sd_ERM

def load_focal(sd):
  return sd

def load_groupdro(sd):
  sd_curr = {key: sd[key] for key in sd.keys() if key != 'q'}
  return sd_curr

def load_jtt(sd):
  sd_curr = {key[10:]: sd[key] for key in sd.keys() if key[:10] == 'cur_model.'}
  return sd_curr

def load_lff(sd):
  sd_curr = {key[11:]: sd[key] for key in sd.keys() if key[:11] == 'pred_model.'}
  return sd_curr
