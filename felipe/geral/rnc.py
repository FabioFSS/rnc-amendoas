# Autor: Fabiano Silva dos Santos
# Universidade Estadual de Santa Cruz - 2021

import os
import keras
from keras import Input
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.layers import Dense, AveragePooling2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
import numpy as np
import csv
from datetime import datetime as dt
import matplotlib.pyplot as plt


# Criando pasta de resultado

# Cria a pasta de resultados caso ela não exista
if not os.path.isdir('results/'):
    os.mkdir('results/')


# Criando a posta do teste que será feito

s_time = dt.now() # Captura o tempo atual

# Cria uma string personalizada com o tempo atual
str_time = f'{str(s_time.year)}-{str(s_time.month)}-{str(s_time.day)}_'
str_time += f'{str(s_time.hour)}-{str(s_time.minute)}-{str(s_time.second)}'

# Cria a pasta do teste
result_dir = f'results/{str(str_time)}/'
os.mkdir(result_dir)


# Definindo hiperparametros

image_size = (376, 356) # Resolução da imagem
image_channels = 3 # Canais da imagem
pool_size = (2, 2) # Tamanho do pooling
strides = (2, 2)
kernel_size = 5
kernel_init = 'random_normal'
bias_init = 'zeros'
padding = 'same' # Tipo de padding das camadas de convolução
activ_func = 'relu' # Função de ativação das camadas
activ_out = 'softmax' # Função de ativação da camada de saída
n_classes = 14 # Número de classes
class_mode = 'categorical'
train_dir = 'dataset/test/' # Pasta do dataset de treino
test_dir = 'dataset/train/' # Pasta do dataset de teste
batch_size = 32 # Tamanho de cada lote
color_mode = 'rgb' # Modo de cor da imagem
epochs = 1500 # Número de épocas de treinamento
learning_rate = 0.001 # Taxa de aprendizado da rede
optimizer = Adam(learning_rate) # Otimizador da rede
loss = 'categorical_crossentropy' # Algoritmo de erro

# Criando string para salvar os parametros no txt
str_hparameters = f'imgsize: {image_size}\nimgchannels: {image_channels}\n'
str_hparameters += f'pool_size: {pool_size}\npadding: {padding}\n'
str_hparameters += f'kernel_size: {kernel_size}\nstrides: {strides}\n'
str_hparameters += f'kernel_init: {kernel_init}\nbias_init: {bias_init}\n'
str_hparameters += f'activ_func: {activ_func}\nactiv_out: {activ_out}\n'
str_hparameters += f'n_classes: {n_classes}\nclass_mode: {class_mode}\n'
str_hparameters += f'train_dir: {train_dir}\ntest_dir: {test_dir}\n'
str_hparameters += f'batch_size: {batch_size}\ncolor_mode: {color_mode}\n'
str_hparameters += f'epochs: {epochs}\nlearning_rate: {learning_rate}\n'
str_hparameters += f'loss: {loss}\n'


# Criando a estrutura da rede

# Cria o modelo rede neural
rede = Sequential() # Cria uma rede sequencial

# Adiciona duas camada de convolução
rede.add(Conv2D(filters=16, kernel_size=kernel_size, strides=strides,
activation=activ_func, padding=padding, input_shape=(*image_size, image_channels),
kernel_regularizer=keras.regularizers.l2(0.01)))
rede.add(BatchNormalization())

rede.add(MaxPooling2D(pool_size=pool_size, padding=padding)) # Adiciona uma camada de Pooling

rede.add(Conv2D(filters=32, kernel_size=kernel_size, strides=strides,
activation=activ_func, padding=padding, input_shape=(*image_size, image_channels),
kernel_regularizer=keras.regularizers.l2(0.01)))
rede.add(BatchNormalization())

rede.add(MaxPooling2D(pool_size=pool_size, padding=padding)) # Adiciona uma camada de Pooling

rede.add(Flatten()) # Adiciona uma camada de achatamento
rede.add(Dense(units=256, activation=activ_func)) # Adiciona uma camada de classificação
rede.add(BatchNormalization())
rede.add(Dropout(0.5)) # Adiciona uma camada de dropout
rede.add(Dense(units=128, activation=activ_func)) # Adiciona uma camada de classificação
rede.add(BatchNormalization())
rede.add(Dropout(0.5)) # Adiciona uma camada de dropout
rede.add(Dense(units=n_classes, activation=activ_out)) # Adiciona uma camada de saída


# Compilando o modelo
rede.compile(optimizer=optimizer, loss=loss, 
metrics=['accuracy'])


# Criando conjunto de treinamento para treino
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.1,
zoom_range=0.1, horizontal_flip=True, vertical_flip=True,
rotation_range=0.1, validation_split=0.2)

training_set = train_datagen.flow_from_directory(train_dir,
target_size=image_size, batch_size=batch_size, class_mode=class_mode,
color_mode=color_mode, subset='training')

validation_set = train_datagen.flow_from_directory(train_dir,
target_size=image_size, batch_size=batch_size, class_mode=class_mode,
color_mode=color_mode, subset='validation')

# Criando conjunto de treinamento para teste
test_datagen = ImageDataGenerator(rescale = 1./255)

test_set = test_datagen.flow_from_directory(test_dir,
target_size=image_size, batch_size=batch_size, class_mode=class_mode,
color_mode=color_mode)

# Criando callbacks

# Cria os critérios de paarada e salvamento do modelo
early_stop = EarlyStopping(monitor='val_accuracy', mode='max',
                            verbose=1, patience=100)
checkpoint = ModelCheckpoint(result_dir+'melhor_modelo.h5', monitor='val_accuracy',
                                mode='max', verbose=1, save_best_only=True)

# Cria o logger
csv_logger = CSVLogger(result_dir+'training_log.csv')

# Adaptando o modelo ao conjunto

historico = rede.fit(training_set, epochs=epochs, validation_data=validation_set,
                    callbacks=[early_stop, checkpoint, csv_logger])


# Avaliando o modelo   
training_loss, training_acc = rede.evaluate(training_set, verbose=1)
validation_loss, validation_acc = rede.evaluate(validation_set, verbose=1)
test_loss, test_acc = rede.evaluate(test_set, verbose=1)

# Modelo final
print(f'\nTraining: loss= {training_loss:.4} acc= {training_acc:.4}')
print(f'\nValidation: loss= {validation_loss:.4} acc= {validation_acc:.4}')
print(f'\nTest: loss= {test_loss:.4} acc= {test_acc:.4}\n')

best_model = load_model(result_dir+'melhor_modelo.h5')

# Avaliando o modelo   
b_training_loss, b_training_acc = best_model.evaluate(training_set, verbose=1)
b_validation_loss, b_validation_acc = best_model.evaluate(validation_set, verbose=1)
b_test_loss, b_test_acc = best_model.evaluate(test_set, verbose=1)

# Melhor modelo
print(f'\nTraining: loss= {training_loss:.4} acc= {training_acc:.4}')
print(f'\nValidation: loss= {validation_loss:.4} acc= {validation_acc:.4}')
print(f'\nTest: loss= {test_loss:.4} acc= {test_acc:.4}\n')


# Salvando a rede
rede.save(result_dir+'modelo_final.h5')


# Lendo arquivo csv
with open(result_dir+'training_log.csv', 'r') as csv_file:
    reader = csv.reader(csv_file, delimiter=',')

    val_accs = []
    val_losses = []

    for row in reader:
        if row[3].replace('.', '').isdigit():
            val_accs.append(float(row[3]))
        
        if row[4].replace('.', '').isdigit():
            val_losses.append(float(row[4]))
    
    max_acc = max(val_accs)
    min_loss = min(val_losses)

# Salva o resumo do modelo e tempo de treino em txt
with open(result_dir+'model.txt', 'w') as file:
    rede.summary(print_fn=lambda x: file.write(x + '\n'))
    file.write(f'\n{str_hparameters}\n')
    file.write(f'\nStart time: {str(s_time)}\n')
    file.write(f'Finish time: {str(dt.now())}\n')
    file.write(f'\nBest:')
    file.write(f'\nTraining: loss= {b_training_loss:.4} acc= {b_training_acc:.4}')
    file.write(f'\nValidation: loss= {b_validation_loss:.4} acc= {b_validation_acc:.4}')
    file.write(f'\nTest: loss= {b_test_loss:.4} acc= {b_test_acc:.4}')
    file.write(f'\nFinal:')
    file.write(f'\nTraining: loss= {training_loss:.4} acc= {training_acc:.4}')
    file.write(f'\nValidation: loss= {validation_loss:.4} acc= {validation_acc:.4}')
    file.write(f'\nTest: loss= {test_loss:.4} acc= {test_acc:.4}')


# Imprimindo um resumo da rede no terminal 
rede.summary()

# Criando os gráficos de acurácia e perda

# Cria a figura e os eixos para o plot 
fig, ax = plt.subplots(2, 1)
fig.suptitle('model accuracy/loss')

# Plota gráfico com as acurácias
ax[0].plot(historico.history['accuracy'])
ax[0].plot(historico.history['val_accuracy'])
ax[0].set(ylabel='accuracy')
ax[0].set(xlabel='epoch')
ax[0].legend(['train', 'val'], loc='upper left')

# Plota gráfico com as perdas
ax[1].plot(historico.history['loss'])
ax[1].plot(historico.history['val_loss'])
ax[1].set(ylabel='loss')
ax[1].set(xlabel='epoch')
ax[1].legend(['train', 'val'], loc='upper left')

# Salva na pasta de resultados do teste
plt.savefig(result_dir+'graphs.png')

# Mostra os gráficos
plt.show()