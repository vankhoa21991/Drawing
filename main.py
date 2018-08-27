from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Flatten, Dropout, TimeDistributed, Reshape, Lambda,Bidirectional
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.optimizers import RMSprop, Adam, SGD
from keras.callbacks import ModelCheckpoint
from utils import *
from data_gen import *

# directory
server = False

if server == True:
    data_dir = "/mnt/DATA/lupin/Dataset/CASIA_extracted/"

else:
    data_dir = '/home/lupin/Cinnamon/Flaxscanner/Dataset/CASIA/Online/Data/preprocessed/'
data_path = 'data/'
run_opt = 1

# load data
stroke_train, stroke_val, label_train, label_val, label2char, char2label = load_data(data_dir)
vocabulary = len(label2char)


# map data to dataloader
batch_size = 8
num_steps = 10000
train_data_generator = DataLoader(stroke_train, label_train,batch_size = batch_size,num_steps=num_steps)
valid_data_generator = DataLoader(stroke_val,label_val,batch_size = batch_size,num_steps=num_steps)


# create model
# define LSTM
hidden_size = 100

model = Sequential()
# model.add(Bidirectional(LSTM(hidden_size, return_sequences=True), input_shape=(317,6)))
# model.add(Dense(vocabulary, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

model.add(LSTM(hidden_size, input_shape=(317,6)))
model.add(Dense(vocabulary, activation='softmax'))
model.compile(loss='categorical_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

print(model.summary())
checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
num_epochs = 50
if run_opt == 1:
    X_train = train_data_generator.pad_strokes
    y_train = to_categorical(train_data_generator.label, num_classes=None)

    model.fit(X_train, y_train, batch_size=batch_size,epochs=100)

    # model.fit_generator(train_data_generator.generate(), 2000, num_epochs,
    #                     validation_data=valid_data_generator.generate(),
    #                     validation_steps=10)
    model.save(data_path + "final_model.hdf5")
elif run_opt == 2:
    model = load_model(data_path + "\model-40.hdf5")
    dummy_iters = 40
    example_training_generator = DataLoader(stroke_train, label_train,batch_size = batch_size,num_steps=num_steps)
    print("Training data:")
    for i in range(dummy_iters):
        dummy = next(example_training_generator.generate())
    num_predict = 10
    true_print_out = "Actual words: "
    pred_print_out = "Predicted words: "
    for i in range(num_predict):
        data = next(example_training_generator.generate())
        prediction = model.predict(data[0])
        predict_word = np.argmax(prediction[:, num_steps-1, :])
        true_print_out += label2char[label_train[num_steps + dummy_iters + i]] + " "
        pred_print_out += label2char[predict_word] + " "
    print(true_print_out)
    print(pred_print_out)




    # test data set
    # dummy_iters = 40
    # example_test_generator = KerasBatchGenerator(test_data, num_steps, 1, vocabulary,
    #                                                  skip_step=1)
    # print("Test data:")
    # for i in range(dummy_iters):
    #     dummy = next(example_test_generator.generate())
    # num_predict = 10
    # true_print_out = "Actual words: "
    # pred_print_out = "Predicted words: "
    # for i in range(num_predict):
    #     data = next(example_test_generator.generate())
    #     prediction = model.predict(data[0])
    #     predict_word = np.argmax(prediction[:, num_steps - 1, :])
    #     true_print_out += label2char[test_data[num_steps + dummy_iters + i]] + " "
    #     pred_print_out += label2char[predict_word] + " "
    # print(true_print_out)
    # print(pred_print_out)
