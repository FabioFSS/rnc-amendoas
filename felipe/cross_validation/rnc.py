import numpy as np
import pandas as pd
import os
import tensorflow as tf
from datetime import datetime as dt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from sklearn.model_selection import KFold, StratifiedKFold
from create_new_model import create_new_model
from matplotlib import pyplot as plt
import csv

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

# Lendo o csv para um dataframe
train_data = pd.read_csv('dataframe/train_labels.csv')
classes_data = train_data[['label']]

# Criando o KFold para a validação cruzada
skf = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)

# Criando o gerador de dados das imagens
idg = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.3,
                         fill_mode='nearest',
                         horizontal_flip = True,
                         rescale=1./255)


# Parametros do treinamento
training_dir = 'dataframe/train'
num_epochs = 500
n_train_samples = len(train_data['filename'])
input_shape = (256, 256, 3)

# Criando string para parametros no txt
str_hparameters = f'Parametros de treinamento da rede:\n'
str_hparameters += f'training_dir: {training_dir}\nresult_dir: {result_dir}\n'
str_hparameters += f'num_epochs: {num_epochs}\nresult_dir: {result_dir}\n'
str_hparameters += f'n_train_samples: {n_train_samples}\n\n'


# Listas que guardaão os valores de precisão e perda dos folds
acc_per_fold = []
loss_per_fold = []

fold_var = 1

for train_index, val_index in skf.split(np.zeros(n_train_samples), classes_data):
	# Salva o tempo de inicio da iteração
	s_time = dt.now()

	# Seleciona a fatia que será usada em cada iteração
	training_data = train_data.iloc[train_index]
	validation_data = train_data.iloc[val_index]
	
	# Cria os sets de treino e validação
	train_set = idg.flow_from_dataframe(training_data, directory=training_dir,
						       x_col='filename', y_col='label',
						       class_mode='categorical', shuffle=True,
							   target_size=input_shape[:2])
	valid_set  = idg.flow_from_dataframe(validation_data, directory=training_dir,
								x_col='filename', y_col='label',
								class_mode='categorical', shuffle=True,
								target_size=input_shape[:2])
	
	# Cria um novo modelo
	model, temp_str_hparameters = create_new_model(input_shape)
	
	# Adiciona os parametros do modelo à string de parametros
	model_str_hparameters = str_hparameters + temp_str_hparameters

	# Cria os callbacks
	model_folder = result_dir+f'model_{fold_var}/'
	os.mkdir(model_folder)
	checkpoint = ModelCheckpoint(model_folder+f'model_{fold_var}.h5', 
							monitor='val_accuracy', verbose=1, 
							save_best_only=True, mode='max')

	early_stop = EarlyStopping(monitor='val_accuracy', mode='max',
                            verbose=1, patience=50)

	csv_logger = CSVLogger(model_folder+'training_log.csv')
	callbacks_list = [checkpoint, early_stop, csv_logger]

	# Treina o modelo
	print(f'\n\nTraining for fold {fold_var}...\n')
	history = model.fit(train_set,
			    epochs=num_epochs,
			    callbacks=callbacks_list,
			    validation_data=valid_set)

	# Avaliando o modelo   
	training_loss, training_acc = model.evaluate(train_set, verbose=1)
	validation_loss, validation_acc = model.evaluate(valid_set, verbose=1)

	# Avaliando no conjunto de validação para o melhor modelo
	model.load_weights(model_folder+f'model_{fold_var}.h5')
	bvalidation_loss, bvalidation_acc = model.evaluate(valid_set, verbose=1)
	acc_per_fold.append(bvalidation_acc)
	loss_per_fold.append(bvalidation_loss)

	# Salva o resumo do modelo e tempo de treino em txt
	with open(model_folder+'model.txt', 'w') as file:
		model.summary(print_fn=lambda x: file.write(x + '\n'))
		file.write(f'\n{model_str_hparameters}\n')
		file.write(f'\nStart time: {str(s_time)}\n')
		file.write(f'Finish time: {str(dt.now())}\n')
		file.write(f'\nBest: loss= {bvalidation_loss:.4} acc= {bvalidation_acc:.4}\n')
		file.write(f'\nFinal:')
		file.write(f'\nTraining: loss= {training_loss:.4} acc= {training_acc:.4}')
		file.write(f'\nValidation: loss= {validation_loss:.4} acc= {validation_acc:.4}\n')
	

	# Criando os gráficos de acurácia e perda

	# Cria a figura e os eixos para o plot 
	fig, ax = plt.subplots(2, 1)
	fig.suptitle('model accuracy/loss')

	# Plota gráfico com as acurácias
	ax[0].plot(history.history['accuracy'])
	ax[0].plot(history.history['val_accuracy'])
	ax[0].set(ylabel='accuracy')
	ax[0].set(xlabel='epoch')
	ax[0].legend(['train', 'val'], loc='upper left')

	# Plota gráfico com as perdas
	ax[1].plot(history.history['loss'])
	ax[1].plot(history.history['val_loss'])
	ax[1].set(ylabel='loss')
	ax[1].set(xlabel='epoch')
	ax[1].legend(['train', 'val'], loc='upper left')

	# Salva na pasta de resultados do teste
	plt.savefig(model_folder+'graphs.png')

	# Incrementa o fold
	fold_var += 1
	input()

with open(result_dir+'final_result.txt', 'w') as result_file:
	string = 'Score per fold:\n'
	for i in range(len(acc_per_fold)):
	 string += f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%\n'
	string += '\nAverage scores for all folds:\n'
	string += f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})\n'
	string += f'> Loss: {np.mean(loss_per_fold)}\n\n'

	result_file.write(string)

	print(f'\n\n{string}')