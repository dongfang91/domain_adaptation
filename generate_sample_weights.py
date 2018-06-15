import read_files as read
import numpy as np
import os
import process_functions as output


def counterList2Dict (counter_list):
    dict_new = dict()
    for item in counter_list:
        dict_new[item[0]]=item[1]
    return dict_new

def hot_vectors2class_index_forweights (labels):
    examples = list()
    n_sen = 0
    for instance in labels:
        n_lable = 0
        label_index = list()
        for label in instance:
            # if list.count(1)==1:
            #     k = list(label).index(1)
            # else:
            #     k = indices = [i for i, x in enumerate(my_list) if x == "whatever"]
            k = list(label).index(1)
            label_index.append(k)
            n_lable +=1

        examples.append(label_index)
        n_sen +=1
    return examples

def create_class_weight(n_labels,labels,mu):
    n_softmax = n_labels
    class_index = hot_vectors2class_index_forweights(labels)
    counts = np.zeros(n_softmax, dtype='int32')
    for softmax_index in class_index:
        softmax_index = np.asarray(softmax_index)
        for i in range(n_softmax):
            counts[i] = counts[i] + np.count_nonzero(softmax_index==i)

    labels_dict = counterList2Dict(list(enumerate(counts, 0)))

    total = np.sum(labels_dict.values())
    class_weight = dict()

    for key, item in labels_dict.items():
        if not item == 0:
            score = mu * total/float(item)
            class_weight[key] = score if score > 1.0 else 1.0
        else:
            class_weight[key] = 10.0

    return class_weight,class_index

def get_sample_weights_multiclass(n_labels,labels,mu1):
    class_weight,class_index = create_class_weight(n_labels,labels,mu=mu1)
    samples_weights = list()
    for instance in class_index:
        sample_weights = [class_weight[category] for category in instance]
        samples_weights.append(sample_weights)
    return samples_weights


def read_newswire():
    char_x_newswire, pos_x_newswire, unicate_x_newswire, vocab_x_newswire = read.load_input("newswire/new_train_4features")
    trainy_interval_newswire = read.load_pos("newswire/char_train_intervalonehotlabels")
    trainy_operator_ex_newswire = read.load_pos("newswire/char_train_exoperatoronehotlabels")
    trainy_operator_im_newswire = read.load_pos("newswire/char_train_imoperatoronehotlabels")
    return char_x_newswire, pos_x_newswire,unicate_x_newswire,trainy_interval_newswire,trainy_operator_ex_newswire,trainy_operator_im_newswire

def read_colon():
    char_x_colon, pos_x_colon, unicate_x_colon = read.load_hdf5("data/colon" + "/train_input", ["char", "pos", "unic"])
    trainy_interval_colon = read.load_hdf5("data/colon" + "/train_output_interval_softmax", ["interval_softmax"])[0]
    trainy_operator_ex_colon = read.load_hdf5("data/colon" + "/train_output_explicit_operator_softmax", ["explicit_operator_softmax"])[0]
    trainy_operator_im_colon = read.load_hdf5("data/colon" + "/train_output_implicit_operator_softmax", ["implicit_operator_softmax"])[0]
    return char_x_colon, pos_x_colon,unicate_x_colon,trainy_interval_colon,trainy_operator_ex_colon,trainy_operator_im_colon


def read_generated_dates():
    syn2_char_x, syn2_pos_x, syn2_unicate_x, syn2_vocab_x = read.load_input("newswire/syn2_training_sentence_input_addmarks3")
    syn2_trainy_interval = read.load_pos("newswire/char_syn_intervalonehotlabels")
    syn2_trainy_operator_ex = read.load_pos("newswire/char_syn_exoperatoronehotlabels")
    syn2_trainy_operator_im = read.load_pos("newswire/char_syn_imoperatoronehotlabels")
    return syn2_char_x, syn2_pos_x, syn2_unicate_x,syn2_trainy_interval,syn2_trainy_operator_ex,syn2_trainy_operator_im

def read_newswire_cv():
    char_x_cv, pos_x_cv, unicate_x_cv, vocab_x_cv = read.load_input("newswire/new_val_4features")
    cv_y_interval = read.load_pos("newswire/char_dev_intervalonehotlabels")
    cv_y_operator_ex = read.load_pos("newswire/char_dev_exoperatoronehotlabels")
    cv_y_operator_im = read.load_pos("newswire/char_dev_imoperatoronehotlabels")
    return char_x_cv,pos_x_cv,unicate_x_cv,cv_y_interval,cv_y_operator_ex,cv_y_operator_im

def save_newswire_cv():
    data_news_cv = read_newswire_cv()
    read.save_hdf5("data/data_mixed/cvnews_train_input", ["char", "pos", "unic"],data_news_cv[:3], ['int8', 'int8', 'int8'])
    read.save_hdf5("data/data_mixed/cvnews_train_target", ["interval_softmax","explicit_operator","implicit_operator"], data_news_cv[3:6], ['int8','int8','int8'])
#save_newswire_cv()

def read_colon_cv():
    char_x_colon, pos_x_colon, unicate_x_colon = read.load_hdf5("data/colon" + "/dev_input", ["char", "pos", "unic"])
    trainy_interval_colon = read.load_hdf5("data/colon" + "/dev_output_interval_softmax", ["interval_softmax"])[0]
    trainy_operator_ex_colon = read.load_hdf5("data/colon" + "/dev_output_explicit_operator_softmax", ["explicit_operator_softmax"])[0]
    trainy_operator_im_colon = read.load_hdf5("data/colon" + "/dev_output_implicit_operator_softmax", ["implicit_operator_softmax"])[0]
    return char_x_colon, pos_x_colon,unicate_x_colon,trainy_interval_colon,trainy_operator_ex_colon,trainy_operator_im_colon

def save_colon_cv():
    data_news_cv = read_colon_cv()
    read.save_hdf5("data/data_mixed/cvcolon_train_input", ["char", "pos", "unic"],data_news_cv[:3], ['int8', 'int8', 'int8'])
    read.save_hdf5("data/data_mixed/cvcolon_train_target", ["interval_softmax","explicit_operator","implicit_operator"], data_news_cv[3:6], ['int8','int8','int8'])
#save_colon_cv()

def get_portion_of_array(dataset,len_data):
    dataset_portion = [data[:len_data] for data in dataset]
    return dataset_portion

def concatenate_multiple_array(newswire,dates,colon):
    combine_data = []
    for index in range(len(newswire)):
        dates_newswire = np.concatenate((dates[index],newswire[index]),axis=0)
        dates_newswire_colon = np.concatenate((dates_newswire,colon[index]),axis=0)
        combine_data.append(dates_newswire_colon)
    return combine_data

def concatenate_binary_array(newswire,dates):
    combine_data = []
    for index in range(len(newswire)):
        dates_newswire = np.concatenate((dates[index],newswire[index]),axis=0)
        combine_data.append(dates_newswire)
    return combine_data

def get_sample_weights(data,mu=0.05):
    sample_weights = []
    for label_set in data:
        sample_weights.append(get_sample_weights_multiclass(label_set.shape[-1],label_set,mu))
    return sample_weights



def generate_portioanl_dataset(portion,len_colon):
    newswire = read_newswire()
    len_newswire = len(newswire[0])
    dates = read_generated_dates()
    len_dates = len(dates[0])
    #len_colon = int(round((len_newswire+len_dates)* float(portion)/(1-float(portion))))
    colon = read_colon()
    #colon_portion = get_portion_of_array(colon,len_colon)
    #data_all = concatenate_multiple_array(newswire,dates,colon_portion)
    #colon_portion_date = concatenate_binary_array(colon_portion, dates)

    ######################################### save news and dates in a new format #####################################
    # news_date = concatenate_binary_array(newswire,dates)
    # read.save_hdf5("data/data_mixed/news_date_train_input", ["char", "pos", "unic"], news_date[:3],['int8', 'int8', 'int8'])
    # read.save_hdf5("data/data_mixed/news_date_train_target", ["interval_softmax", "explicit_operator", "implicit_operator"], news_date[3:6], ['int8', 'int8', 'int8'])
    # sample_weights_news_date = get_sample_weights(news_date[3:6], mu=0.05)
    # np.save("data/data_mixed/news_date_sample_weights", sample_weights_news_date)

     ###############################   save news in a new format ########################################################
    read.save_hdf5("data/data_mixed/news_train_input", ["char", "pos", "unic"], newswire[:3],['int8', 'int8', 'int8'])
    read.save_hdf5("data/data_mixed/news_train_target", ["interval_softmax", "explicit_operator", "implicit_operator"], newswire[3:6], ['int8', 'int8', 'int8'])
    sample_weights_news_date = get_sample_weights(newswire[3:6], mu=0.05)
    np.save("data/data_mixed/news_sample_weights", sample_weights_news_date)



    ######################################### save dates + news + n% colon   and save  n% colon#####################################

    # read.save_hdf5("data/data_mixed/colon_" + str(portion) + "_train_input", ["char", "pos", "unic"],colon_portion[:3], ['int8', 'int8', 'int8'])
    # read.save_hdf5("data/data_mixed/news_colon_"+str(portion)+"_train_input", ["char","pos","unic"], data_all[:3], ['int8','int8','int8'])
    #read.save_hdf5("data/data_mixed/colon_portion_date_" + str(portion) + "_train_input", ["char", "pos", "unic"], colon_portion_date[:3],['int8', 'int8', 'int8'])


    # read.save_hdf5("data/data_mixed/colon_" + str(portion) + "_train_target", ["interval_softmax", "explicit_operator", "implicit_operator"],colon_portion[3:6], ['int8', 'int8', 'int8'])
    # read.save_hdf5("data/data_mixed/news_colon_"+str(portion)+"_train_target", ["interval_softmax","explicit_operator","implicit_operator"], data_all[3:6], ['int8','int8','int8'])
    #read.save_hdf5("data/data_mixed/colon_portion_date_" + str(portion) + "_train_target", ["interval_softmax","explicit_operator","implicit_operator"], colon_portion_date[3:6], ['int8', 'int8', 'int8'])

    #
    # sample_weights_colon = get_sample_weights(colon_portion[3:6],mu=0.05)
    # sample_weights_all = get_sample_weights(data_all[3:6],mu=0.05)
    #sample_weights_colon_portion_date = get_sample_weights(colon_portion_date[3:6], mu=0.05)

    # np.save("data/data_mixed/colon_" + str(portion) + "_sample_weights",sample_weights_colon)
    # np.save("data/data_mixed/news_colon_"+str(portion) + "_sample_weights", sample_weights_all)
    #np.save("data/data_mixed/colon_portion_date_" + str(portion) + "_sample_weights", sample_weights_colon_portion_date)

    print portion

generate_portioanl_dataset(portion="1_9",len_colon = 19100)

####### whole scate entities:
####### newswire: 1620
####### colon:    1619    1837 sentences  1
####### colon:    6470    8020 sentences  4
####### colon:    9710    11960 sentences  6
####### colon:   14741    19100 sentences  9



def calculate_amount_of_entities(len_colon):
    #newswire = read_newswire()
    colon = read_colon()
    colon_portion = get_portion_of_array(colon, len_colon)
    amounts = 0
    for output_target in colon_portion[3:6]:
        output_target_index = hot_vectors2class_index_forweights(output_target)
        class_loc = output.found_location_with_constraint(output_target_index)
        span = output.loc2span(class_loc, None, post_process=False)
        for sentence_span in span:
            if len(sentence_span[0])>0:
                amounts += len(sentence_span)
    print amounts




#calculate_amount_of_entities(19100)


