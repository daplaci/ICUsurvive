import os
from keras.models import Model
from keras.layers import Concatenate, Dense,Embedding, LSTM, Input, BatchNormalization
from keras.layers import CuDNNLSTM, GRU, CuDNNGRU, Dropout, Concatenate, Reshape
from keras import backend as K
from keras.regularizers import l2
import nnet_survival


stripper = lambda x : x[0] if ((type(x) is tuple or type(x) is list) and len(x)==1)  else x

#dictionary of the class
#it return a dictionary {name of the function to call : specific compartment} #for ehr this is not used anymore
def class_register(cls):
    cls._propdict = {}
    for methodname in dir(cls):
        method = getattr(cls, methodname)
        if '_prop' in dir(method):
            print (method, methodname)
            cls._propdict.update(
                {methodname: method._prop})
    return cls

# register the methods that build the model
def register(*args):
    def wrapper(func):
        func._prop = args
        return func
    return wrapper

@class_register
class ModelConstructor():
    def __init__(self, args, best_model_name):
        self.args = args
        self.use_gpu = bool(K.tensorflow_backend._get_available_gpus()) and args.use_gpu
        self.recurrent_args = {'units':self.args.units, 'return_sequences':self.args.apply_post_attention}
        if args.recurrent_layer =='lstm':
            if self.use_gpu:    
                self.recurrent_layer = CuDNNLSTM
            else:
                self.recurrent_args['activation'] = self.args.activation
                self.recurrent_args['recurrent_dropout'] = self.args.recurrent_dropout
                self.recurrent_layer = LSTM
        if args.recurrent_layer =='gru':
            if self.use_gpu:    
                self.recurrent_layer = CuDNNGRU
            else:
                self.recurrent_args['activation'] = self.args.activation
                self.recurrent_args['recurrent_dropout'] = self.args.recurrent_dropout
                self.recurrent_layer = GRU
        
        self.best_model_name = best_model_name

    def get_feat_embedding(self, feat_obj,):
        input_layer = Input(shape=(feat_obj.inputs.shape[1:]), name='input_{}'.format(feat_obj.feat_name))
        if feat_obj.feat_name in ['ageadm', 'los_before_icu']:
             return input_layer, input_layer
        
        embedding = Embedding(feat_obj.vocab_size, feat_obj.emb_size)(input_layer)
        pool = Reshape((self.args.padd_time, -1),)(embedding)
        pool = Dropout(self.args.dropout)(pool)
        pool = Dense(feat_obj.emb_size,kernel_regularizer=l2(self.args.l2_regularizer), bias_regularizer=l2(self.args.l2_regularizer), name='dense_emb_{}'.format(feat_obj.feat_name))(pool)
        pool = BatchNormalization()(pool)
        return input_layer, pool

    @register('ehr')
    def ehr_model(self,inputs):

        input_layers, pools = map(list, zip(*[self.get_feat_embedding(feat_obj) for feat_obj in inputs]))

        input_lstm = Concatenate(-1)(pools)
        input_lstm = Dropout(self.args.dropout)(input_lstm)

        output_lstm = self.recurrent_layer(**self.recurrent_args)(input_lstm)
        
        prediction = Dense(len(self.args.window_list), activation='sigmoid', kernel_regularizer=l2(self.args.l2_regularizer), bias_regularizer=l2(self.args.l2_regularizer))(output_lstm)

        model = Model(inputs=input_layers, outputs=prediction)
        model.compile(loss=nnet_survival.surv_likelihood(len(self.args.window_list)), optimizer=self.args.optimizer) 
        return model

    def load_weights(self, model, compartment):
        params = self.best_model_name.split('.')
        for idx, param in enumerate(params):
            if 'compartment' in param:
                params[idx] = 'compartment_' + compartment
        weights_to_load = '.'.join(params) + '.hdf5'
        if weights_to_load not in os.listdir(self.args.output_dir  + "/best_weights/"):
            raise Exception("There is a model in the ensemble that cannot find any weights to load.\nWeights not found : ", weights_to_load)
        model.load_weights(self.args.output_dir  + "/best_weights/" + weights_to_load)
        return model

# this function returns a keras model built in the class ModelConstractur with the method labelled with the specific compartment
def get_model(inputs, args, best_model_name=None):
    
    model_constructor = ModelConstructor(args, best_model_name)
    if type(args.compartment) is list:
        model_label = 'merged'
    else:
        model_label = stripper(args.compartment)
      
    for method, methods_prop in model_constructor._propdict.items():
        if model_label in methods_prop:
            model = getattr(model_constructor, method)(inputs)
            return model
            
    raise Exception("No model found for the model type ", args.compartment)

def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations