import numpy as np
import h5py
np.random.seed(20170915)
from keras.layers.wrappers import Bidirectional
from keras.layers import Dense

from keras.layers import Embedding, LSTM, Input, Lambda, Concatenate
from keras.callbacks import CSVLogger
from keras.regularizers import l1, l2
from keras.models import Model,load_model
import keras.backend as K
from keras.callbacks import ModelCheckpoint,Callback
import os



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


def trainging(storage,exp,sampleweights,char_x,trainy_interval,trainy_operator_ex,trainy_operator_im,
            batchsize,epoch_size,reload = False,modelpath = None):

    seq_length = char_x.shape[1]
    type_size_interval = trainy_interval.shape[-1]
    type_size_operator_ex = trainy_operator_ex.shape[-1]
    type_size_operator_im = trainy_operator_im.shape[-1]

    if not os.path.exists(storage):
        os.makedirs(storage)
    if reload == False:


        char_input = Input(shape=(seq_length,), dtype='int8', name='character')
        forward_embedding_layer = Embedding(input_dim=227, output_dim=100)(char_input)
        forward_lstm_layer = LSTM(2048, return_sequences=True, recurrent_activation='sigmoid')(forward_embedding_layer)

        backward_embedding_layer = Embedding(input_dim=227, output_dim=100)(char_input)
        backward_lstm_layer = LSTM(2048, return_sequences=True, recurrent_activation='sigmoid', go_backwards=True)(backward_embedding_layer)
        reversed_backward_lstm_layer = Lambda(lambda tensor: K.reverse(tensor, axes=1),output_shape=(356,2048))(backward_lstm_layer)
        merged_lstm_layers = Concatenate(axis=2)([forward_lstm_layer, reversed_backward_lstm_layer])

        interval_output = Dense(type_size_interval, activation='softmax', kernel_regularizer=l2(.01), name='dense_1')(merged_lstm_layers)

        explicit_operator = Dense(type_size_operator_ex, activation='softmax', kernel_regularizer=l2(.01),
                                  name='dense_2') (merged_lstm_layers)

        implicit_operator = Dense(type_size_operator_im, activation='softmax', kernel_regularizer=l2(.01),
                                  name='dense_3') (merged_lstm_layers)

        model = Model(inputs=[char_input],
                      outputs=[interval_output, explicit_operator, implicit_operator])

        model.compile(optimizer='rmsprop',
                      loss={'dense_1': 'categorical_crossentropy',
                            'dense_2': 'categorical_crossentropy',
                            'dense_3': 'categorical_crossentropy'},
                      loss_weights={'dense_1': 1.0, 'dense_2': 0.75, 'dense_3': 0.5},
                      metrics=['categorical_accuracy'],
                      sample_weight_mode="temporal")

        model_flair = load_model("data/flair_keras" + ".h5")
        model.layers[2].set_weights(model_flair.layers[2].get_weights())
        model.layers[1].set_weights(model_flair.layers[1].get_weights())
        model.layers[4].set_weights(model_flair.layers[4].get_weights())
        model.layers[3].set_weights(model_flair.layers[3].get_weights())

    else:
        model = load_model("model_files/" + modelpath + ".hdf5")
    print(model.summary())
    filepath = storage + "/weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=False)
    # es_performance = Es_performance(filepath, [char_x_cv, pos_x_cv,unicate_x_cv],golds,gold_locs,
    #         filename='training_real_%s.csv' % exp,verbose = 0,min_delta=0.005, patience=50,es_target=target)
    csv_logger = CSVLogger('training_%s.csv' % exp)
    callbacks_list = [checkpoint,csv_logger]  # ,es_performance]

    hist = model.fit(x ={'character': char_x},
                     y={'dense_1': trainy_interval, 'dense_2': trainy_operator_ex,'dense_3': trainy_operator_im}, epochs=epoch_size,
                     batch_size=batchsize, callbacks=callbacks_list,sample_weight=sampleweights)
    model.save(storage + '/model_result.hdf5')



file_path ="data"  #"/extra/dongfangxu9/domain_adaptation/data/data_mixed"

portion = str(1)
char_x= load_hdf5(file_path + "/news_date_train_input_flair", ["char"])[0]


trainy_interval,trainy_operator_ex,trainy_operator_im = load_hdf5(file_path + "/news_date_train_target", ["interval_softmax","explicit_operator","implicit_operator"])

sampleweights = list(np.load(file_path+"/news_date_sample_weights.npy"))


epoch_size = 800
batchsize = 10
path = "data/model/"  #"/xdisk/dongfangxu9/domain_adaptation/"

exp = "flair_news"
target = 3
storage = path + exp


trainging(storage,exp,sampleweights,char_x,trainy_interval,trainy_operator_ex,trainy_operator_im,
            batchsize,epoch_size,reload = False,modelpath = None)