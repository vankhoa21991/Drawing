from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda,Bidirectional
from keras.layers import LSTM, AveragePooling1D
from keras.layers import LSTM, Input

from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import *
from data_gen import *

def recognition_model(args,vocabulary):
    
    # This returns a tensor
    _input = Input(shape=(317, 6))

    x = Bidirectional(LSTM(args.hidden_size, recurrent_dropout=0.3), merge_mode='ave')(_input)

    x = Dropout(0.2)(x)
    x = Reshape((args.hidden_size,1))(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = Reshape((int(args.hidden_size/2),))(x)
    x = Dropout(0.2)(x)
    x = Dense(200)(x)

    predictions = Dense(vocabulary, activation='softmax')(x)
    
    model = Model(inputs=_input, outputs=predictions)
    
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    print(model.summary())
    return model

def train(args):
    # load model
    #stroke_train, stroke_val, label_train, label_val, label2char, char2label,max_len = load_data(args.data_dir,args.model_dir)
    
    data = np.load(args.data_dir + args.data_filename, encoding='latin1')
    stroke_train = list(data['train'])
    label_train = list(data['label_train'])
    stroke_val = list(data['valid'])
    label_val = list(data['label_val'])

    max_len = data['max_len']


    # load encode file
    char2label = json.load(open(args.model_dir + 'encode_kanji.json'))
    label2char = {}
    for k, v in char2label.items():
        label2char[v] = k
    
    vocabulary = len(label2char)

    train_data = DataLoader(stroke_train, label_train, args=args)
    valid_data = DataLoader(stroke_val, label_val, args=args)

    if args.is_resume:
        model = load_model(args.model_dir + "final_model.hdf5")
    else:
        model = recognition_model(args, vocabulary)

    X_train = train_data.pad_strokes
    y_train = to_categorical(train_data.label, num_classes=vocabulary)
    X_val = valid_data.pad_strokes
    y_val = to_categorical(valid_data.label, num_classes=vocabulary)

    history = model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.num_epochs,
              validation_split=0.2)
    
    model.save(args.model_dir + "final_model.hdf5")
    
    # list all model in history
    print(history.history.keys())
    # summarize history for accuracy
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 
    ax.plot(history.history['acc'])
    ax.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig.savefig(args.sample_dir + 'acc.png') 
    
    fig, ax = plt.subplots( nrows=1, ncols=1 ) 
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    fig.savefig(args.sample_dir + 'loss.png')


def evaluate(args):
    #stroke_train, stroke_val, label_train, label_val, label2char, char2label, max_len = load_data(args.data_dir,args.model_dir)
    
    
    data = np.load(args.data_dir + args.data_filename, encoding='latin1')
    stroke_test = list(data['test'])
    label_test = list(data['label_test'])

    max_len = data['max_len']

    # load encode file
    char2label = json.load(open(args.model_dir + 'encode_kanji.json'))
    label2char = {}
    for k, v in char2label.items():
        label2char[v] = k
    
    vocabulary = len(label2char)
    test_data = DataLoader(stroke_test, label_test, args=args)

    model = load_model(args.model_dir + "final_model.hdf5")

    true_print_out = "Actual words: "
    pred_print_out = "Predicted words: "
    index = random.sample(range(0,len(test_data.pad_strokes)), 1000)
    data = test_data.pad_strokes[index, :,:]
    prediction = model.predict(data)

    for i in range(len(index)):
        predict_word = np.argmax(prediction[i, :])
        true_print_out += label2char[test_data.label[index[i]]] + " "
        pred_print_out += label2char[predict_word] + " "
    print(true_print_out)
    print(pred_print_out)
    
    X_test = test_data.pad_strokes[index, :,:]
    y_test_ = [test_data.label[index[i]] for i in range(len(index))]
    y_test = to_categorical(y_test_, num_classes=vocabulary)
    
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # environment
    server = True

    if server == True:
        parser.add_argument('--data_dir', default='/mnt/DATA/lupin/Flaxscanner/Dataset/CASIA/Drawing/')
        parser.add_argument('--data_filename', default='casia_preprocessed_20.npz')
        parser.add_argument('--model_dir', default='/mnt/DATA/lupin/Flaxscanner/Models/Drawing/recog_model/')
        parser.add_argument('--sample_dir', default='/mnt/DATA/lupin/Flaxscanner/Drawing/recog_model/samples/')
    else:
        parser.add_argument('--data_dir', default='model/')
        parser.add_argument('--model_dir', default='model/')
        

    parser.add_argument('--mode', default='test', type=str)
    parser.add_argument('--num_epochs', default=120, type=int)
    parser.add_argument('--hidden_size', default=500, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--max_seq_length', default=317, type=int)

    parser.add_argument('--batch_size', default=600, type=int)
    parser.add_argument('--num_gpu', default='1', type=int)
    parser.add_argument('--is_resume', default=False, type=bool)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.num_gpu)
    
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)