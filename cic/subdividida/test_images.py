import matplotlib.pyplot as plt
import numpy as np
import os 
import shutil
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing.image import img_to_array, save_img
from keras.models import load_model

def testar(rede, path, target_size, show=False):
    '''Recebe um caminho e executa o teste da rede neural para a imagem.
    rede = uma rede do keras.
    path = caminho da imagem.
    show = True para mostrar a imagem testada, False para não mostrar.
    '''

    img = load_img(path, target_size=target_size) # Carrega a imagem
    img = img_to_array(img) # Converte a imagem para vetor
    img = img/255 # Muda a escala para valores de 0 a 1, que foi utilizado no treino
    img = np.expand_dims(img, axis=0) # Expande o número de dimensões
    result = np.argmax(rede.predict(img), axis=-1) # Testa a imagem na rede

    # Mostra a imagem caso show = True
    if show:
        plt.imshow(img) # Determina que a imagem será mostrada
        plt.title(str(result[0])) # Determina o título da imagem
        plt.show() # Mostra a imagem

    return int(result) # Retorna o indice da classe como um valor inteiro

date1 = '2021-6-9_1-33-31/'
date2 = '2021-6-9_4-50-48/'
date3 = '2021-6-9_5-47-12/'

result_path1 = 'net_results/first-pass/results/' + date1 # Diretório de da pasta dos resultados
result_path2 = 'net_results/second-pass/results/' + date2 # Diretório de da pasta dos resultados
result_path3 = 'net_results/third-pass/results/' + date3 # Diretório de da pasta dos resultados

net_path1 = result_path1 + 'melhor_modelo.h5' # Caminho para a rede neural
net_path2 = result_path2 + 'melhor_modelo.h5' # Caminho para a rede neural
net_path3 = result_path3 + 'melhor_modelo.h5' # Caminho para a rede neural

errors_path = 'images_tests/errors/'+ date1 # O diretório da pasta dos erros
# Verifica se a pasta de erros já existe, exclui caso exista
if os.path.isdir(errors_path):
    shutil.rmtree(errors_path)

# Cria uma pasta de erros caso não exista
if not os.path.isdir(errors_path):
    os.makedirs(errors_path)


target_size = (120, 230)

rede1 = load_model(net_path1) # Carrega o modelo
rede2 = load_model(net_path2) # Carrega o modelo
rede3 = load_model(net_path3) # Carrega o modelo

classes1 = {0: 'Marrom', 1: 'Outros', 2: 'Violeta1', 3: 'Violeta2'}
classes2 = {0: 'Achatada', 1: 'Ardosia', 2: 'BCompart', 3: 'DanifInsetos', 4: 'Outros', 5: 'Quebradas'}
classes3 = {0: 'BChapada', 1: 'Dupla', 2: 'Germinada', 3: 'Mofo'}

dataset_path = 'images_tests/test_dataset/'
folders = os.listdir(dataset_path)
acc_per_class = [[0, 0] for folder in folders]

answers = 0
wrong_answers = 0
for folder in folders:
    folder_path = dataset_path + folder + '/'
    images = os.listdir(folder_path)

    for image in images:
        image_path = folder_path + image

        result = testar(rede1, image_path, target_size)
        result_str = classes1[result]

        if result_str == 'Outros':
            result = testar(rede2, image_path, target_size)
            result_str = classes2[result]
        
            if result_str == 'Outros':
                result = testar(rede3, image_path, target_size)
                result_str = classes3[result]
            
        if result_str != folder:
            # Verifica se a pasta da classificação existe, se não cria
            if not os.path.isdir(errors_path+result_str):
                os.mkdir(errors_path+result_str)

            # Copia a imagem para a pasta em que ela foi classificada
            img = load_img(image_path)
            save_img(errors_path+result_str+'/'+image, img)
            wrong_answers += 1
            acc_per_class[folders.index(folder)][1] += 1

        answers += 1
        acc_per_class[folders.index(folder)][0] += 1

string = ''
for acc in acc_per_class:
    class_acc = (acc[0] - acc[1])/acc[0]
    string += f'Precisao para - {folders[acc_per_class.index(acc)]}: {class_acc}\n'

accuracy = (answers - wrong_answers)/answers
string += f'\nPrecisão final: {accuracy}'

with open(errors_path + 'results.txt', 'w') as file:
    file.write(string)