from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda,Bidirectional
from keras.layers import LSTM, AveragePooling1D
from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint
from utils import *
from data_gen import *





def Model(args,vocabulary):
    model = Sequential()
    model.add(Bidirectional(LSTM(args.hidden_size), input_shape=(317, 6)))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=200))
    model.add(Dense(output_dim=vocabulary, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model

def train(args):
    # load data
    stroke_train, stroke_val, label_train, label_val, label2char, char2label = load_data(args.data_dir,args.model_dir)
    vocabulary = len(label2char)

    train_data_generator = DataLoader(stroke_train, label_train, args=args)
    valid_data_generator = DataLoader(stroke_val, label_val, args=args)

    if args.is_resume:
        model = load_model(args.model_dir + "final_model.hdf5")
    else:
        model = Model(args, vocabulary)

    X_train = train_data_generator.pad_strokes
    y_train = to_categorical(train_data_generator.label, num_classes=vocabulary)
    X_val = valid_data_generator.pad_strokes
    y_val = to_categorical(valid_data_generator.label, num_classes=vocabulary)

    model.fit(X_train, y_train, batch_size=args.batch_size, epochs=args.num_epochs,
              validation_data=(X_val,y_val))
    

    model.save(args.model_dir + "final_model.hdf5")

def evaluate(args):
    model = load_model(args.model_dir + "\model-40.hdf5")
    dummy_iters = 40
    example_training_generator = DataLoader(stroke_train, label_train, args=args)

    num_predict = 10
    true_print_out = "Actual words: "
    pred_print_out = "Predicted words: "
    for i in range(num_predict):
        data = next(example_training_generator.generate())
        prediction = model.predict(data[0])
        predict_word = np.argmax(prediction[:, args.num_steps - 1, :])
        true_print_out += label2char[label_train[args.num_steps + dummy_iters + i]] + " "
        pred_print_out += label2char[predict_word] + " "
    print(true_print_out)
    print(pred_print_out)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # environment
    server = False

    if server == True:
        parser.add_argument('--data_dir', default='/mnt/DATA/lupin/Dataset/CASIA_extracted/')
        parser.add_argument('--model_dir', default='/mnt/DATA/lupin/Drawing/keras_model/data/')
    else:
        parser.add_argument('--data_dir', default='/home/lupin/Cinnamon/Flaxscanner/Dataset/CASIA/Online/Data/preprocessed/')
        parser.add_argument('--model_dir', default='data/')

    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--hidden_size', default=500, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--max_seq_length', default=317, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--is_resume', default=False, type=bool)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'

    train(args)