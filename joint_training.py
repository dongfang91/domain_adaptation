import numpy as np
import os
import math
import h5py
import json

from collections import OrderedDict
from collections import Iterable
import csv
np.random.seed(20170915)
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense, TimeDistributed, merge
from keras.layers import GRU, Input,LSTM
from keras.layers import Embedding
from keras.callbacks import CSVLogger
from keras.regularizers import l1, l2
from keras.models import Model,load_model
from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint,Callback
from keras.callbacks import LearningRateScheduler




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



def trainging(storage,exp,sampleweights,char_x,pos_x,unicate_x,trainy_interval,trainy_operator_ex,trainy_operator_im,
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


    else:
        model = load_model("/xdisk/dongfangxu9/time_expression/new_sentence_level/char/"+modelpath+".hdf5")



    filepath = storage + "/weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='fmeasure', verbose=1, save_best_only=False)
    #es_performance = Es_performance(filepath, [char_x_cv, pos_x_cv,unicate_x_cv],golds,gold_locs,
    #        filename='training_real_%s.csv' % exp,verbose = 0,min_delta=0.8, patience=10,es_target=target)
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
char_x,pos_x,unicate_x = load_hdf5(file_path + "/news_colon_1_"+portion+"_train_input", ["char", "pos", "unic"])
char_x_cv,pos_x_cv,unicate_x_cv= load_hdf5(file_path + "/cvnews_train_input", ["char", "pos", "unic"])

trainy_interval,trainy_operator_ex,trainy_operator_im = load_hdf5(file_path + "/news_colon_1_"+portion+"_train_target", ["interval_softmax","explicit_operator","implicit_operator"])
cv_y_interval,cv_y_operator_ex,cv_y_operator_im = load_hdf5(file_path + "/cvnews_train_target", ["interval_softmax","explicit_operator","implicit_operator"])



n_pos = 49
n_char = 89
n_unicate = 15
n_vocab = 16
epoch_size = 800
batchsize = 110



sampleweights = list(np.load(file_path+"/news_colon_1_"+portion+"_sample_weights.npy"))

#print sample_weights1.shape
#print sample_weights.shape
path = "/xdisk/dongfangxu9/domain_adaptation"

exp = "/news_colon_1_"+portion+"_devon_news"
storage = path + exp
trainging(storage,exp,sampleweights,char_x,pos_x,unicate_x,trainy_interval,trainy_operator_ex,trainy_operator_im,
                                        char_x_cv,pos_x_cv,unicate_x_cv,cv_y_interval,cv_y_operator_ex,cv_y_operator_im,batchsize,epoch_size,
                                        n_char,n_pos,n_unicate,n_vocab,reload = False,modelpath = "med_3softmax_5_29_pretrainedmodel/weights-improvement-231",embedding_size_char =128,
                                        embedding_size_pos = 32, embedding_size_unicate = 64,embedding_size_vocab =16,
                                        gru_size1 =256,gru_size2 = 150)
