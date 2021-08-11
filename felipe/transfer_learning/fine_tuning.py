from keras.models import load_model, save_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import keras

image_size = (256, 256) # Resolução da imagem
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


model = load_model('results/2021-5-4_14-53-47/modelo_final.h5')

# model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),  # Low learning rate
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.CategoricalAccuracy()],
)

epochs = 50
model.fit(training_set, epochs=epochs, validation_data=validation_set)
save_model('results/2021-5-4_14-53-47/modelo_ajuste_fino.h5')