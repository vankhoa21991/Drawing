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
import matplotlib.pyplot as plt
from utils import *
from data_gen import *

def recognition_model(args,vocabulary):
    
    # This returns a tensor
    _input = Input(shape=(317, 6))

    x = Bidirectional(LSTM(args.hidden_size))(_input)

    x = Dropout(0.2)(x)
    x = Reshape((args.hidden_size*2,1))(x)
    x = AveragePooling1D(pool_size=2)(x)
    x = Reshape((args.hidden_size,))(x)
    x = Dropout(0.2)(x)
    x = Dense(200)(x)
    x = Dropout(0.2)(x)

    predictions = Dense(vocabulary, activation='softmax')(x)
    
    model = Model(inputs=_input, outputs=predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model

def train(args):
    # load data
    stroke_train, stroke_val, label_train, label_val, label2char, char2label,max_len = load_data(args.data_dir,args.model_dir)
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

    model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.num_epochs,
              validation_data=(X_val,y_val))

    # list all data in history
    print(model.history.keys())
    # summarize history for accuracy
    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    model.save(args.model_dir + "final_model.hdf5")

def evaluate(args):
    stroke_train, stroke_val, label_train, label_val, label2char, char2label, max_len = load_data(args.data_dir,args.model_dir)
    vocabulary = len(label2char)
    test_data = DataLoader(stroke_val, label_val, args=args)

    model = load_model(args.model_dir + "final_model.hdf5")



    true_print_out = "Actual words: "
    pred_print_out = "Predicted words: "
    index = random.sample(range(0,len(test_data.pad_strokes)), 100)
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
    server = False

    if server == True:
        parser.add_argument('--data_dir', default='/mnt/DATA/lupin/Drawing/keras_model/data/')
        parser.add_argument('--model_dir', default='/mnt/DATA/lupin/Drawing/keras_model/data/')
    else:
        parser.add_argument('--data_dir', default='data/')
        parser.add_argument('--model_dir', default='data/')
        

    parser.add_argument('--mode', default='test', type=str)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--hidden_size', default=500, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--max_seq_length', default=317, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--is_resume', default=False, type=bool)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)