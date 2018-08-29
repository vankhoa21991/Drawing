from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda,Bidirectional
<<<<<<< HEAD
from keras.layers import LSTM, AveragePooling1D
=======
from keras.layers import LSTM, Input
>>>>>>> ecea335f30ecadf2019d77b3d16b84619917d370
from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint
from utils import *
from data_gen import *





<<<<<<< HEAD
def Model(args,vocabulary):
    model = Sequential()
    model.add(Bidirectional(LSTM(args.hidden_size), input_shape=(317, 6)))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=200))
    model.add(Dense(output_dim=vocabulary, activation='softmax'))
=======
def recognition_model(args,vocabulary):
    
    # This returns a tensor
    _input = Input(shape=(317, 6))

    x = Bidirectional(LSTM(args.hidden_size))(_input)
    x = Dropout(0.1)(x)
    predictions = Dense(input_dim = 200, output_dim=vocabulary, activation='softmax')(x)
    
    model = Model(inputs=_input, outputs=predictions)

>>>>>>> ecea335f30ecadf2019d77b3d16b84619917d370
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model

def train(args):
    # load data
    stroke_train, stroke_val, label_train, label_val, label2char, char2label = load_data(args.data_dir,args.model_dir)
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
    
    model.save(args.model_dir + "final_model.hdf5")

def evaluate(args):
    stroke_train, stroke_val, label_train, label_val, label2char, char2label = load_data(args.data_dir,args.model_dir)
    vocabulary = len(label2char)
    
    model = load_model(args.model_dir + "final_model.hdf5")
    dummy_iters = 40
    
    test_data = DataLoader(stroke_val, label_val, args=args)

    true_print_out = "Actual words: "
    pred_print_out = "Predicted words: "
    for i in range(random.randint(0,len(test_data.pad_strokes))):
        data = test_data.pad_strokes[:,i,:]
        prediction = model.predict(test_data.pad_strokes)
        predict_word = np.argmax(prediction[:, args.max_seq_length - 1, :])
        true_print_out += label2char[label_train[i]] + " "
        pred_print_out += label2char[predict_word] + " "
    print(true_print_out)
    print(pred_print_out)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # environment
    server = True

    if server == True:
        parser.add_argument('--data_dir', default='/mnt/DATA/lupin/Dataset/CASIA_extracted/')
        parser.add_argument('--model_dir', default='/mnt/DATA/lupin/Drawing/keras_model/data/')
    else:
        parser.add_argument('--data_dir', default='/home/lupin/Cinnamon/Flaxscanner/Dataset/CASIA/Online/Data/preprocessed/')
        parser.add_argument('--model_dir', default='data/')
        
    parser.add_argument('--mode', default='test', type=str)
    parser.add_argument('--num_epochs', default=40, type=int)
    parser.add_argument('--hidden_size', default=500, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--max_seq_length', default=317, type=int)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--is_resume', default=False, type=bool)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    
    if args.mode == 'train':
        train(args)
    else:
        evaluate(args)