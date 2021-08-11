# Autor: Fabiano Silva dos Santos
# Universidade Estadual de Santa Cruz - 2021

import os
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.layers import Dense, AveragePooling2D, Dropout, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
import numpy as np
import csv
from datetime import datetime as dt
import matplotlib.pyplot as plt
from keras.applications.inception_resnet_v2 import InceptionResNetV2 as Inception
from keras.models import Model


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

image_size = (200, 200) # Resolução da imagem
image_channels = 3 # Canais da imagem
activ_func = 'relu' # Função de ativação das camadas
activ_out = 'softmax' # Função de ativação da camada de saída
kernel_init = 'he_normal'
n_classes = 14 # Número de classes
class_mode = 'categorical'
train_dir = 'dataset/train/' # Pasta do dataset de treino
test_dir = 'dataset/test/' # Pasta do dataset de teste
batch_size = 32 # Tamanho de cada lote
color_mode = 'rgb' # Modo de cor da imagem
epochs = 1500 # Número de épocas de treinamento
learning_rate = 0.001 # Taxa de aprendizado da rede
optimizer = Adam(learning_rate) # Otimizador da rede
loss = 'categorical_crossentropy' # Algoritmo de erro

# Criando string para salvar os parametros no txt
str_hparameters = f'imgsize: {image_size}\nimgchannels: {image_channels}\n'
str_hparameters += f'kernel_init: {kernel_init}\n'
str_hparameters += f'activ_func: {activ_func}\nactiv_out: {activ_out}\n'
str_hparameters += f'n_classes: {n_classes}\nclass_mode: {class_mode}\n'
str_hparameters += f'train_dir: {train_dir}\ntest_dir: {test_dir}\n'
str_hparameters += f'batch_size: {batch_size}\ncolor_mode: {color_mode}\n'
str_hparameters += f'epochs: {epochs}\nlearning_rate: {learning_rate}\n'
str_hparameters += f'loss: {loss}\n'


# Criando a estrutura da rede
# Carregando modelo base
base_model = Inception(input_shape=(*image_size, 3), include_top=False, weights='imagenet')
x = base_model.output
base_model.trainable = False

# Introduz novas camadas
x = GlobalAveragePooling2D()(x)
dense_layers = Dense(256, activation=activ_func, kernel_initializer=kernel_init)(x)
dense_layers = Dropout(0.3)(dense_layers)
out_layer = Dense(14, activation=activ_out)(dense_layers)

# Cria modelo final
rede = Model(inputs=[base_model.input], outputs=[out_layer], name='rede_final')

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