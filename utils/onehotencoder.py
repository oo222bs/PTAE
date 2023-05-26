from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from numpy import array
import numpy as np
import os

def encode_descriptions(desc_file, target_folder, num_of_samp, max_len, vocab_size, dates):
    file = open(desc_file, 'r')
    corpus = file.read().splitlines()
    doc = []
    for i in range(len(corpus)): #range(17549):#
        corpus[i] = "<BOS> " + corpus[i] + " <EOS>"  #corpus = "<BOS> " + "<EOS>"
        corpus[i] = corpus[i].lower().split()
        #corpus = corpus.lower().split()
        #for j in range(max_len-len(corpus[i])):
        #    corpus[i] = corpus[i] + ["<pad>"]
        doc = doc + corpus[i]
        #doc = doc + corpus
    values = array(doc)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    onehot_encoded = np.reshape(onehot_encoded, (num_of_samp,max_len,vocab_size))
    onehot = np.concatenate((onehot_encoded[:1440], onehot_encoded[1440:2880],onehot_encoded[2880:4320], onehot_encoded[4320:5760]), axis=1)

    file2 = open('../actionstcnoexc.txt', 'r')
    acts = file2.read().splitlines()
    test = []
    for i, act in enumerate(acts):
        if act.split(' ')[-1] == "test":
            test.append(i)
    test = np.loadtxt(target_folder + 'test.txt')

    for date in dates:
        for i in range(len(onehot)):
            description = onehot
            dirname = ''
            dirname_v = ''
            save_name = "target" + str(i).zfill(6) + '.txt'
            if i in test:
                dirname = target_folder + "language_test/" + date+ '/'
            else:
                dirname = target_folder + "language_train/" + date+ '/'
            save_name = os.path.join(dirname, save_name)
            if not os.path.exists(os.path.dirname(save_name)):
                os.makedirs(os.path.dirname(save_name))
            np.savetxt(save_name, description[i], fmt="%.6f")

if __name__ == "__main__":
    encode_descriptions('../descriptionstcnoexc.txt', '../target_three_cubes_no_exc/', 1440, 4, 22, ["230526"])
