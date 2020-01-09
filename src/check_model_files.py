from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

model = load_model('simple_CNN.530-0.65.hdf5')


print(model.summary())

#### flattened list
##for layer in model.layers:
##    print(layer)
##    print('\n')
