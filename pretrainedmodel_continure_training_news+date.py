import numpy as np
np.random.seed(20170915)
import os
import math
import h5py
import json

from collections import OrderedDict
from collections import Iterable
import csv

from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, TimeDistributed, merge
from keras.layers import GRU, Input
from keras.layers import Embedding
from keras.models import load_model
from keras.callbacks import CSVLogger
from keras.regularizers import l1, l2
from keras.models import Model,load_model
from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint,Callback
from keras.callbacks import LearningRateScheduler




def load_input(filename):
    with h5py.File('data/'+ filename + '.hdf5', 'r') as hf:
        print("List of arrays in this file: \n", hf.keys())
        x1 = hf.get('char')
        x2 = hf.get('pos')
        x3 = hf.get('unic')
        x4 = hf.get('vocab')



        x_char = np.array(x1)
        x_pos = np.array(x2)
        x_unic = np.array(x3)
        x_vocab = np.array(x4)

        #n_patterns = x_data.shape[0]

        # print x_data[0][1000:1100]
        # print y_data[0][1000:1100]
        #y_data = y_data.reshape(y_data.shape+(1,))
    del x1,x2,x4,x3
    return x_char,x_pos,x_unic,x_vocab


def load_pos(filename):
    with h5py.File('data/' + filename + '.hdf5', 'r') as hf:
        print("List of arrays in this file: \n", hf.keys())
        x = hf.get('input')

        x_data = np.array(x)
        # n_patterns = x_data.shape[0]

        # print x_data[0][1000:1100]
        # print y_data[0][1000:1100]
        # y_data = y_data.reshape(y_data.shape+(1,))
        print(x_data.shape)
    del x
    return x_data


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

def trainging(model_trained,storage,exp,sampleweights,char_x,pos_x,unicate_x,trainy_interval,trainy_operator_ex,trainy_operator_im,
                                        char_x_cv,pos_x_cv,unicate_x_cv,cv_y_interval,cv_y_operator_ex,cv_y_operator_im,batchsize,epoch_size,
                                        n_char,n_pos,n_unicate,n_vocab,reload = False,modelpath = None,embedding_size_char =64,
                                        embedding_size_pos = 48, embedding_size_unicate = 32,embedding_size_vocab =32,
                                        gru_size1 = 128,gru_size2 = 160):

    seq_length = char_x.shape[1]
    type_size_interval = trainy_interval.shape[-1]
    type_size_operator_ex = trainy_operator_ex.shape[-1]
    type_size_operator_im = trainy_operator_im.shape[-1]



    if not os.path.exists(storage):
        os.makedirs(storage)
    if reload ==False:

        char_input = Input(shape=(seq_length,), dtype='int8', name='character')
        char_em = Embedding(output_dim=embedding_size_char, input_dim=n_char, input_length=seq_length,
                            W_regularizer = l2(.01),mask_zero=True,dropout = 0.25)(char_input)

        pos_input = Input(shape=(seq_length,), dtype='int8', name='pos')
        pos_em = Embedding(output_dim=embedding_size_pos, input_dim=n_pos, input_length=seq_length,
                           W_regularizer=l2(.01), mask_zero=True, dropout=0.15)(pos_input)

        unicate_input = Input(shape=(seq_length,), dtype='int8', name='unicate')
        unicate_em = Embedding(output_dim=embedding_size_unicate, input_dim=n_unicate, input_length=seq_length,
                           W_regularizer=l2(.01), mask_zero=True, dropout=0.15)(unicate_input)

        #vocab_input = Input(shape=(seq_length,), dtype='int8', name='vocab')
        #vocab_em = Embedding(output_dim=embedding_size_vocab, input_dim=n_vocab, input_length=seq_length,
        #                     W_regularizer=l1(.01), mask_zero=True, dropout=0.1)(vocab_input)

        #input_merge = merge([char_em,pos_em,unicate_em,vocab_em], mode='concat')
        input_merge = merge([char_em, pos_em, unicate_em], mode='concat')

        gru_out_1 = Bidirectional(GRU(gru_size1, input_shape=(seq_length, embedding_size_char+embedding_size_pos+embedding_size_unicate),
                                      return_sequences=True))(input_merge)

        gru_out_2 = GRU(gru_size2, return_sequences=True) (gru_out_1)

        interval_output = TimeDistributed(Dense(type_size_interval, activation='softmax',W_regularizer = l2(.01),name='timedistributed_1'))(gru_out_2)

        gru_out_3 = Bidirectional(GRU(gru_size1, input_shape=(seq_length, embedding_size_char+embedding_size_pos+embedding_size_unicate),
                                      return_sequences=True))(input_merge)

        #gru_out_3_1 = GRU(gru_size2, return_sequences=True) (gru_out_3)
        gru_out_4 = GRU(gru_size2, return_sequences=True)(gru_out_3)

        explicit_operator = TimeDistributed(Dense(type_size_operator_ex, activation='softmax',W_regularizer = l2(.01),name='timedistributed_2'))(gru_out_4)

        gru_out_5 = Bidirectional(GRU(gru_size1, input_shape=(seq_length, embedding_size_char+embedding_size_pos+embedding_size_unicate),
                                      return_sequences=True))(input_merge)

        gru_out_6 = GRU(gru_size2, return_sequences=True) (gru_out_5)

        implicit_operator = TimeDistributed(Dense(type_size_operator_im, activation='softmax', W_regularizer=l2(.01), name='timedistributed_3'))(gru_out_6)


        ####model = Model(input=[char_input, pos_input,unicate_input,vocab_input], output=[interval_output, explicit_operator,implicit_operator])
        model = Model(input=[char_input, pos_input,unicate_input],
                      output=[interval_output, explicit_operator, implicit_operator])
        #lrate = LearningRateScheduler(step_decay)

        #rmsprop = RMSprop(lr=0.0012)

        model.compile(optimizer='rmsprop',
                      loss={'timedistributed_1': 'categorical_crossentropy', 'timedistributed_2': 'categorical_crossentropy','timedistributed_3': 'categorical_crossentropy'},
                      loss_weights={'timedistributed_1': 1., 'timedistributed_2': 0.75,'timedistributed_3': 0.5},metrics=['fmeasure', 'categorical_accuracy'],
                      sample_weight_mode="temporal")

        char_weights = model_trained.layers[3].get_weights()
        pos_weights = model_trained.layers[4].get_weights()
        unicate_weights = model_trained.layers[5].get_weights()

        bigru1 = model_trained.layers[7].get_weights()
        bigru2 = model_trained.layers[8].get_weights()
        bigru3 = model_trained.layers[9].get_weights()

        gru1 = model_trained.layers[10].get_weights()
        gru2 = model_trained.layers[11].get_weights()
        gru3 = model_trained.layers[12].get_weights()


        model.layers[3].set_weights(char_weights)
        model.layers[4].set_weights(pos_weights)
        model.layers[5].set_weights(unicate_weights)
        model.layers[7].set_weights(bigru1)
        model.layers[8].set_weights(bigru2)
        model.layers[9].set_weights(bigru3)
        model.layers[10].set_weights(gru1)
        model.layers[11].set_weights(gru2)
        model.layers[12].set_weights(gru3)

    else:
        model = load_model("model_files/"+modelpath+".hdf5")



    filepath = storage + "/weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='fmeasure', verbose=1, save_best_only=False)
    # es_performance = Es_performance(filepath, [char_x_cv, pos_x_cv,unicate_x_cv],golds,gold_locs,
    #         filename='training_real_%s.csv' % exp,verbose = 0,min_delta=0.005, patience=50,es_target=target)
    csv_logger = CSVLogger('training_%s.csv' % exp)
    callbacks_list = [checkpoint, csv_logger]#,es_performance]


    hist = model.fit({'character': char_x, 'pos': pos_x,'unicate':unicate_x},
                  {'timedistributed_1': trainy_interval, 'timedistributed_2': trainy_operator_ex,'timedistributed_3': trainy_operator_im}, nb_epoch=epoch_size,
                  batch_size=batchsize, callbacks=callbacks_list,validation_data =({'character': char_x_cv, 'pos': pos_x_cv,'unicate':unicate_x_cv},
                  {'timedistributed_1': cv_y_interval, 'timedistributed_2': cv_y_operator_ex,'timedistributed_3':cv_y_operator_im}),sample_weight=sampleweights)
    model.save(storage + '/model_result.hdf5')
    np.save(storage + '/epoch_history.npy', hist.history)




file_path = "/extra/dongfangxu9/domain_adaptation/data/data_mixed"

portion = str(1)
char_x,pos_x,unicate_x = load_hdf5(file_path + "/news_date_train_input", ["char", "pos", "unic"])
char_x_cv,pos_x_cv,unicate_x_cv= load_hdf5(file_path + "/cvnews_train_input", ["char", "pos", "unic"])

trainy_interval,trainy_operator_ex,trainy_operator_im = load_hdf5(file_path + "/news_date_train_target", ["interval_softmax","explicit_operator","implicit_operator"])
cv_y_interval,cv_y_operator_ex,cv_y_operator_im = load_hdf5(file_path + "/cvnews_train_target", ["interval_softmax","explicit_operator","implicit_operator"])




n_pos = 49
n_char = 89
n_unicate = 15
n_vocab = 16
epoch_size = 800
batchsize = 100


sampleweights = list(np.load(file_path+"/news_date_sample_weights.npy"))

#print sample_weights1.shape
#print sample_weights.shape
path = "/xdisk/dongfangxu9/domain_adaptation/"

exp = "pretrainedon_colon_1_"+portion+"_traingingon_news+dates_devon_news"
target = 3
storage = path + exp

model_trained = load_model("/extra/dongfangxu9/domain_adaptation/data/models/colon_1_"+portion+"._weights-improvement-799.hdf5")

trainging(model_trained,storage,exp,sampleweights,char_x,pos_x,unicate_x,trainy_interval,trainy_operator_ex,trainy_operator_im,
                                        char_x_cv,pos_x_cv,unicate_x_cv,cv_y_interval,cv_y_operator_ex,cv_y_operator_im,batchsize,epoch_size,
                                        n_char,n_pos,n_unicate,n_vocab,reload = False,modelpath = "med_3softmax_5_29_pretrainedmodel/weights-improvement-231",embedding_size_char =128,
                                        embedding_size_pos = 32, embedding_size_unicate = 64,embedding_size_vocab =16,
                                        gru_size1 =256,gru_size2 = 150)
