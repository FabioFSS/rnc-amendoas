import matplotlib.pyplot as plt
import numpy as np
import os 
import shutil
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing.image import img_to_array, save_img
from keras.models import load_model

def testar(rede, path, show=False):
    '''Recebe um caminho e executa o teste da rede neural para a imagem.
    rede = uma rede do keras.
    path = caminho da imagem.
    show = True para mostrar a imagem testada, False para não mostrar.
    '''

    img = load_img(path, target_size=image_size) # Carrega a imagem
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

result_path = 'results/2021-6-1_23-17-37/' # Diretório de da pasta dos resultados
net_path = result_path + 'melhor_modelo.h5' # Caminho para a rede neural
testset_path = 'dataset/test/' # O diretório do conjunto de teste
errors_path = result_path + 'erros/' # O diretório da pasta dos erros
image_size = (120, 230) # O tamanho da imagem
class_mode = 'categorical' # O modo de classe

rede = load_model(net_path) # Carrega o modelo
rede.summary() # Imprime no terminal um resumo da rede

test_datagen = ImageDataGenerator(rescale = 1./255) # Cria uma gerador de dados

# Cria o conjunto de teste
test_set = test_datagen.flow_from_directory(testset_path,
target_size=image_size, class_mode=class_mode)

# Avaliando o modelo   
print('\nAvaliando a rede:')
test_loss, test_acc = rede.evaluate(test_set, verbose=1)

# Verifica se a pasta de erros já existe, exclui caso exista
if os.path.isdir(errors_path):
    shutil.rmtree(errors_path)

# Cria uma pasta de erros caso não exista
if not os.path.isdir(errors_path):
    os.mkdir(errors_path)

folders = os.listdir(testset_path) # Lista as pastas das classes que serão testadas

total_errors = 0 # Contador de erros
total_answers = 0 # Contador de respostas totais
errors_per_class = []

# Itera sobre cada pasta das classes
print('\nTestando imagens individualmente...')
for folder in folders:

    errors_per_class.append([folder, 0])
    images = os.listdir(testset_path+folder) # Lista os arquivos (imagens) da pasta
    
    # Itera sobre cada imagem da lista de imagens
    for image in images:
        
        result = testar(rede, testset_path+folder+'/'+image) # Testa a imagem na rede
        total_answers += 1 # Incrementa 1 ao número total de testes

        # Verifica se o valor da resposta é errado
        if result != folders.index(folder):
            
            total_errors += 1 # Incrementa o número de erros
            errors_per_class[folders.index(folder)][1] += 1 # Armazena o número de erros por classe

            # Verifica se a pasta da classificação existe, se não cria
            if not os.path.isdir(errors_path+folders[result]):
                os.mkdir(errors_path+folders[result])

            # Copia a imagem para a pasta em que ela foi classificada
            img = load_img(testset_path+folder+'/'+image)
            save_img(errors_path+folders[result]+'/'+image, img)
            

# Imprime no terminal os resultados
errors_str = f'Dados de erros geral\n'
errors_str += f'Total de respostas: {total_answers}\n'
errors_str += f'Erros: {total_errors}\n'
errors_str += f'Acertos: {total_answers-total_errors}\n'
errors_str += f'Precisao: {(total_answers-total_errors)/total_answers}\n\n'

# Adiciona os dados de erro por classe à string
errors_str += f'Dados de erro por classe\n'
images_per_class = (total_answers/len(errors_per_class))
for image_class in errors_per_class:
    image_class.append((images_per_class-image_class[1])/images_per_class) # Adiciona a precisão por classe
    errors_str += f'{image_class[0]}: erros={image_class[1]}    precisao={image_class[2]}\n' 

print('\n\n'+errors_str)

# Salva os resultados do teste em um txt na pasta de erros
with open(errors_path+'test_results.txt', 'w+') as file:
    file.write(errors_str)