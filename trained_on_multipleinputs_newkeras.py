import numpy as np
import os
import h5py

np.random.seed(20170915)
from keras.layers.wrappers import Bidirectional,TimeDistributed
from keras.layers import GRU, Dropout, Embedding, Dense,Input
from keras.regularizers import l2
from keras.regularizers import Regularizer
from keras.models import Model
from keras.callbacks import CSVLogger,ModelCheckpoint
import keras
import keras.backend as K

from keras.callbacks import ModelCheckpoint,Callback

def load_hdf5(filename,labels):
    data = list()
    with h5py.File(filename + '.hdf5', 'r') as hf:
        print("List of datum in this file: ", hf.keys())
        for label in labels:
            x = hf.get(label)
            x_data = np.array(x)
            del x
            print("The shape of datum "+ label +": ",x_data.shape)
            data.append(x_data)
    return data


class Changeable_loss(Callback):
    """Callback that records events into a `History` object.

    This callback is automatically applied to
    every Keras model. The `History` object
    gets returned by the `fit` method of models.
    """

    def __init__(self, alpha_news, beta_news,gamma_news,alpha_colon, beta_colon,gamma_colon):
        self.alpha_news = alpha_news
        self.beta_news = beta_news
        self.gamma_news = gamma_news
        self.alpha_colon = alpha_colon
        self.beta_colon = beta_colon
        self.gamma_colon = gamma_colon

    # def on_epoch_begin(self, epoch, logs=None):


    # customize your behavior
    def on_batch_begin(self, batch, logs={}):
        if batch <5:
            K.set_value(self.alpha_news, 1.0)
            K.set_value(self.beta_news, 0.75)
            K.set_value(self.gamma_news, 0.5)
            K.set_value(self.alpha_colon, 0.0)
            K.set_value(self.beta_colon, 0.0)
            K.set_value(self.gamma_colon, 0.0)
            set_model_l1_l2(self.model,l1 =0,l2 =0.0,n_layer=23)
            set_model_l1_l2(self.model, l1=0, l2=0.0, n_layer=26)
        elif batch >=5:
            K.set_value(self.alpha_news, 0.0)
            K.set_value(self.beta_news, 0.0)
            K.set_value(self.gamma_news, 0.0)
            K.set_value(self.alpha_colon, 1.0)
            K.set_value(self.beta_colon, 0.75)
            K.set_value(self.gamma_colon, 0.50)
            set_model_l1_l2(self.model,l1 =0,l2 =0.0,n_layer=23)
            set_model_l1_l2(self.model, l1=0, l2=0.0, n_layer=26)



class L1L2_m(Regularizer):
    """Regularizer for L1 and L2 regularization.
    https://github.com/keras-team/keras/issues/4813
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0.0, l2=0.01):
        with K.name_scope(self.__class__.__name__):
            self.l1 = K.variable(l1, name='l1')
            self.l2 = K.variable(l2, name='l2')
            self.val_l1 = l1
            self.val_l2 = l2

    def set_l1_l2(self, l1, l2):
        K.set_value(self.l1, l1)
        K.set_value(self.l2, l2)
        self.val_l1 = l1
        self.val_l2 = l2

    def __call__(self, x):
        regularization = 0.
        if self.val_l1 > 0.:
            regularization += K.sum(self.l1 * K.abs(x))
        if self.val_l2 > 0.:
            regularization += K.sum(self.l2 * K.square(x))
        return regularization

    def get_config(self):
        config = {'l1': float(K.get_value(self.l1)),
                  'l2': float(K.get_value(self.l2))}
        return config

from keras.utils.generic_utils import get_custom_objects

get_custom_objects().update({ L1L2_m.__name__: L1L2_m })


def set_model_l1_l2(model,l1,l2,n_layer):
    for layer in model.layers[n_layer:n_layer+3]:
         if 'kernel_regularizer' in dir(layer) and isinstance(layer.kernel_regularizer, L1L2_m):
                layer.kernel_regularizer.set_l1_l2(l1,l2)


def trainging(storage,exp,sampleweights,char_x_news,pos_x_news,unicate_x_news,char_x_colon,pos_x_colon,unicate_x_colon,trainy_interval_news,
                                        trainy_operator_ex_news,trainy_operator_im_news,trainy_interval_colon,trainy_operator_ex_colon,trainy_operator_im_colon,
                                        char_x_cv,pos_x_cv,unicate_x_cv,cv_y_interval,cv_y_operator_ex,cv_y_operator_im,batchsize,epoch_size,
                                        n_char,n_pos,n_unicate,n_vocab,reload = False,modelpath = None,embedding_size_char =64,
                                        embedding_size_pos = 48, embedding_size_unicate = 32,embedding_size_vocab =32,
                                        gru_size1 = 128,gru_size2 = 160):

    seq_length = char_x_news.shape[1]
    type_size_interval = trainy_interval_news.shape[-1]
    type_size_operator_ex = trainy_operator_ex_news.shape[-1]
    type_size_operator_im = trainy_operator_im_news.shape[-1]



    if not os.path.exists(storage):
        os.makedirs(storage)

    CharEmbedding = Embedding(output_dim=embedding_size_char, input_dim=n_char, input_length=seq_length,
                        embeddings_regularizer = l2 (.01),mask_zero=True)

    PosEmbedding = Embedding(output_dim=embedding_size_pos, input_dim=n_pos, input_length=seq_length,
                        embeddings_regularizer = l2 (.01),mask_zero=True)

    UnicateEmbedding = Embedding(output_dim=embedding_size_unicate, input_dim=n_unicate, input_length=seq_length,
                        embeddings_regularizer = l2(.01),mask_zero=True)

    Gru_out_1 = Bidirectional(GRU(gru_size1,return_sequences=True,input_shape=(seq_length, embedding_size_char+embedding_size_pos+embedding_size_unicate)))
    Gru_out_2 = GRU(gru_size2, return_sequences=True)

    Interval_output_news = Dense(type_size_interval, activation='softmax', kernel_regularizer=L1L2_m(l2=0.0), name='dense_1')
    Interval_output_colon = Dense(type_size_interval, activation='softmax', kernel_regularizer=L1L2_m(l2=0.0), name='dense_4')

    Gru_out_3 = Bidirectional(GRU(gru_size1,return_sequences=True))
    Gru_out_4 = GRU(gru_size2, return_sequences=True)

    Explicit_operator_news = Dense(type_size_operator_ex, activation='softmax', kernel_regularizer=L1L2_m(l2=0.0), name='dense_2')
    Explicit_operator_colon = Dense(type_size_operator_ex, activation='softmax', kernel_regularizer=L1L2_m(l2=0.0), name='dense_5')

    Gru_out_5 = Bidirectional(GRU(gru_size1,return_sequences=True))
    Gru_out_6 = GRU(gru_size2, return_sequences=True)

    Implicit_operator_news = Dense(type_size_operator_im, activation='softmax', kernel_regularizer=L1L2_m(l2=0.0), name='dense_3')
    Implicit_operator_colon = Dense(type_size_operator_im, activation='softmax', kernel_regularizer=L1L2_m(l2=0.0), name='dense_6')

    char_input_news = Input(shape=(seq_length,), dtype='int8', name='character_news')
    char_input_colon = Input(shape=(seq_length,), dtype='int8', name='character_colon')

    pos_input_news = Input(shape=(seq_length,), dtype='int8', name='pos_news')
    pos_input_colon = Input(shape=(seq_length,), dtype='int8', name='pos_colon')

    unicate_input_news = Input(shape=(seq_length,), dtype='int8', name='unicate_news')
    unicate_input_colon = Input(shape=(seq_length,), dtype='int8', name='unicate_colon')

    char_em_news  = Dropout(0.25)(CharEmbedding(char_input_news))
    pos_em_news = Dropout(0.15)(PosEmbedding(pos_input_news))
    unicate_em_news = Dropout(0.15)(UnicateEmbedding(unicate_input_news))

    char_em_colon  = Dropout(0.25)(CharEmbedding(char_input_colon ))
    pos_em_colon  = Dropout(0.15)(PosEmbedding(pos_input_colon ))
    unicate_em_colon  = Dropout(0.15)(UnicateEmbedding(unicate_input_colon ))

    merged_news = keras.layers.concatenate([char_em_news,pos_em_news,unicate_em_news],axis=-1)
    merged_colon = keras.layers.concatenate([char_em_colon, pos_em_colon, unicate_em_colon], axis=-1)

    gru_out1_news = Gru_out_1(merged_news)
    gru_out2_news = Gru_out_2(gru_out1_news)
    interval_output_news = Interval_output_news(gru_out2_news)

    gru_out3_news = Gru_out_3(merged_news)
    gru_out4_news = Gru_out_4(gru_out3_news)
    explicit_operator_news = Explicit_operator_news(gru_out4_news)

    gru_out5_news = Gru_out_5(merged_news)
    gru_out6_news = Gru_out_6(gru_out5_news)
    implicit_operator_news = Implicit_operator_news(gru_out6_news)

    gru_out1_colon = Gru_out_1(merged_colon)
    gru_out2_colon = Gru_out_2(gru_out1_colon)
    interval_output_colon = Interval_output_colon(gru_out2_colon)

    gru_out3_colon = Gru_out_3(merged_colon)
    gru_out4_colon = Gru_out_4(gru_out3_colon)
    explicit_operator_colon = Explicit_operator_colon(gru_out4_colon)

    gru_out5_colon = Gru_out_5(merged_colon)
    gru_out6_colon = Gru_out_6(gru_out5_colon)
    implicit_operator_colon = Implicit_operator_colon(gru_out6_colon)

    model = Model(inputs=[char_input_news, pos_input_news, unicate_input_news, char_input_colon, pos_input_colon,
                         unicate_input_colon],
                  outputs=[interval_output_news, explicit_operator_news, implicit_operator_news, interval_output_colon,
                          explicit_operator_colon, implicit_operator_colon])

    alpha_news = K.variable(1.00)
    beta_news = K.variable(0.75)
    gamma_news = K.variable(0.5)
    alpha_colon = K.variable(0.0)
    beta_colon = K.variable(0.0)
    gamma_colon = K.variable(0.0)

    model.compile(optimizer='sgd',
                  loss={'dense_1': 'categorical_crossentropy',
                        'dense_2': 'categorical_crossentropy',
                        'dense_3': 'categorical_crossentropy',
                        'dense_4': 'categorical_crossentropy',
                        'dense_5': 'categorical_crossentropy',
                        'dense_6': 'categorical_crossentropy',
                        },
                  loss_weights={'dense_1': alpha_news, 'dense_2': beta_news, 'dense_3': gamma_news,'dense_4': alpha_colon, 'dense_5': beta_colon, 'dense_6': gamma_colon},
                  metrics=['categorical_accuracy'],
                  sample_weight_mode="temporal")
    print(model.summary())




    filepath = storage + "/weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False)
    csv_logger = CSVLogger('training_%s.csv' % exp)
    changeable_loss = Changeable_loss(alpha_news, beta_news, gamma_news,alpha_colon,beta_colon,gamma_colon)
    callbacks_list = [checkpoint, csv_logger,changeable_loss]

    hist = model.fit(x ={'character_news': char_x_news, 'pos_news': pos_x_news,'unicate_news':unicate_x_news,'character_colon': char_x_colon,
                         'pos_colon': pos_x_colon,'unicate_colon':unicate_x_colon}, y={'dense_1': trainy_interval_news, 'dense_2': trainy_operator_ex_news,
                     'dense_3': trainy_operator_im_news,'dense_4': trainy_interval_colon, 'dense_5': trainy_operator_ex_colon,'dense_6': trainy_operator_im_colon},
                     epochs=epoch_size,batch_size=batchsize, callbacks=callbacks_list,shuffle=False)#,validation_data =({'character_news': char_x_cv, 'pos_news': pos_x_cv,'unicate_news':unicate_x_cv},
                    #{'dense_1': cv_y_interval, 'dense_2': cv_y_operator_ex, 'dense_3': cv_y_operator_im}),)#,sample_weight=sampleweights)
    model.save(storage + '/model_result.hdf5')
    np.save(storage + '/epoch_history.npy', hist.history)





# ##################################folder for saving the input files###################################
# #file_path = "/extra/dongfangxu9/domain_adaptation/data/data_mixed"
# #
file_path= "data/data_mixed"

portion = str(1)
char_x_news,pos_x_news,unicate_x_news = load_hdf5(file_path + "/news_train_input", ["char", "pos", "unic"])
#char_x_cv,pos_x_cv,unicate_x_cv= load_hdf5(file_path + "/cvnews_train_input", ["char", "pos", "unic"])
char_x_colon,pos_x_colon,unicate_x_colon = load_hdf5(file_path + "/cvcolon_train_input", ["char", "pos", "unic"])

char_x_null_news = np.zeros(char_x_colon.shape)
pos_x_null_news = np.zeros(pos_x_colon.shape)
unicate_x_null_news = np.zeros(unicate_x_colon.shape)
char_x_null_news[:, 0] = 1
pos_x_null_news[:, 0] = 1
unicate_x_null_news[:, 0] = 1

char_x_null_colon = np.zeros(char_x_news.shape)
pos_x_null_colon = np.zeros(pos_x_news.shape)
unicate_x_null_colon = np.zeros(unicate_x_news.shape)
char_x_null_colon[:, 0] = 1
pos_x_null_colon[:, 0] = 1
unicate_x_null_colon[:, 0] = 1


char_x_news = np.concatenate((char_x_news[:50],char_x_null_news[:50]))
pos_x_news = np.concatenate((pos_x_news[:50],pos_x_null_news[:50]))
unicate_x_news = np.concatenate((unicate_x_news[:50],unicate_x_null_news[:50]))
char_x_colon = np.concatenate((char_x_null_colon[:50],char_x_colon[:50]))
pos_x_colon = np.concatenate((pos_x_null_colon[:50],char_x_null_news[:50]))
unicate_x_colon = np.concatenate((unicate_x_null_colon[:50],char_x_null_news[:50]))



trainy_interval_news,trainy_operator_ex_news,trainy_operator_im_news = load_hdf5(file_path + "/news_train_target", ["interval_softmax","explicit_operator","implicit_operator"])
trainy_interval_colon,trainy_operator_ex_colon,trainy_operator_im_colon = load_hdf5(file_path + "/cvcolon_train_target", ["interval_softmax","explicit_operator","implicit_operator"])

trainy_interval_null_news = np.zeros(trainy_interval_colon.shape)
trainy_operator_ex_null_news = np.zeros(trainy_operator_ex_colon.shape)
trainy_operator_im_null_news = np.zeros(trainy_operator_im_colon.shape)
trainy_interval_null_news[:, 0] = 1
trainy_operator_ex_null_news[:, 0] = 1
trainy_operator_im_null_news[:, 0] = 1

trainy_interval_null_colon = np.zeros(trainy_interval_news.shape)
trainy_operator_ex_null_colon = np.zeros(trainy_operator_ex_news.shape)
trainy_operator_im_null_colon = np.zeros(trainy_operator_im_news.shape)
trainy_interval_null_colon[:, 0] = 1
trainy_operator_ex_null_colon[:, 0] = 1
trainy_operator_im_null_colon[:, 0] = 1

trainy_interval_news = np.concatenate((trainy_interval_news[:50],trainy_interval_null_news[:50]))
trainy_operator_ex_news = np.concatenate((trainy_operator_ex_news[:50],trainy_operator_ex_null_news[:50]))
trainy_operator_im_news = np.concatenate((trainy_operator_im_news[:50],trainy_operator_im_null_news[:50]))

trainy_interval_colon = np.concatenate((trainy_interval_null_colon[:50],trainy_interval_colon[:50]))
trainy_operator_ex_colon = np.concatenate((trainy_operator_ex_null_colon[:50],trainy_operator_ex_colon[:50]))
trainy_operator_im_colon = np.concatenate((trainy_operator_im_null_colon[:50],trainy_operator_im_colon[:50]))



#cv_y_interval,cv_y_operator_ex,cv_y_operator_im = load_hdf5(file_path + "/cvnews_train_target", ["interval_softmax","explicit_operator","implicit_operator"])


n_pos = 49
n_char = 89
n_unicate = 15
n_vocab = 16
epoch_size = 800
batchsize = 10



sampleweights = list(np.load(file_path+"/news_sample_weights.npy"))


#################################################path to save the output files and csv log files #########################################
path = "/xdisk/dongfangxu9/domain_adaptation/"

exp = "news_devon_news_newkeras"
storage = path + exp
trainging(storage,exp,sampleweights,char_x_news,pos_x_news,unicate_x_news, char_x_colon , pos_x_colon,unicate_x_colon ,trainy_interval_news ,
          trainy_operator_ex_news,trainy_operator_im_news,trainy_interval_colon,trainy_operator_ex_colon,trainy_operator_im_colon,
          None,None,None,None,None,None,
          batchsize,epoch_size,n_char,n_pos,n_unicate,n_vocab,reload = False,modelpath = "med_3softmax_5_29_pretrainedmodel/weights-improvement-231",
          embedding_size_char =128,embedding_size_pos = 32, embedding_size_unicate = 64,embedding_size_vocab =16, gru_size1 =256,gru_size2 = 150)

# from keras.models import load_model
# path = "/xdisk/dongfangxu9/domain_adaptation/"
# model_trained = load_model(path + "news_devon_news_newkeras/weights-improvement-01.hdf5")
# print(1)