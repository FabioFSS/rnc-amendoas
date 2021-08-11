from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.layers import Dropout, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

def create_new_model(input_shape):
    # Definindo hiperparametros
    
    # Parametros da estrutura da rede
    input_shape = input_shape
    conv_activ = 'relu'
    den_activ = 'relu'
    out_activ = 'softmax'
    kernel_size = 5
    strides = (2, 2)
    pool_size = (2, 2)
    padding = 'same'
    n_classes = 14
    kernel_init = 'random_normal'
    bias_init = 'zeros'

    # Parametros de compilação da rede
    optimizer = Adam(0.001)
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']

    # Criando string para salvar os parametros no txt
    str_hparameters = f'Parametros da estrutura da rede:\n'
    str_hparameters += f'input_shape: {input_shape}\n'
    str_hparameters += f'conv_activ: {conv_activ}\nden_activ: {den_activ}\n'
    str_hparameters += f'out_activ: {out_activ}\npool_size: {pool_size}\n'
    str_hparameters += f'kernel_size: {kernel_size}\strides: {strides}\n'
    str_hparameters += f'padding: {padding}\n_classes: {n_classes}\n\n'

    str_hparameters += f'Parametros de compilação da rede:\n'
    str_hparameters += f'optimizer: {optimizer}\nloss: {loss}\n'
    str_hparameters += f'metrics: {metrics}\n\n'

    # Cria o modelo rede neural
    model = Sequential() # Cria uma rede sequencial

    # Adiciona duas camada de convolução
    model.add(Conv2D(filters=32, kernel_size=kernel_size, strides=strides, activation=conv_activ,
    padding=padding, input_shape=input_shape,
    kernel_regularizer=l2(0.001), kernel_initializer=kernel_init,
    bias_initializer=bias_init))

    model.add(MaxPooling2D(pool_size=pool_size, padding=padding)) # Adiciona uma camada de Pooling

    model.add(Conv2D(filters=32, kernel_size=kernel_size, strides=strides, activation=conv_activ,
    padding=padding, input_shape=input_shape,
    kernel_regularizer=l2(0.001), kernel_initializer=kernel_init,
    bias_initializer=bias_init))

    model.add(MaxPooling2D(pool_size=pool_size, padding=padding)) # Adiciona uma camada de Pooling

    model.add(Flatten()) # Adiciona uma camada de achatamento
    model.add(Dense(units=1024, activation=den_activ)) # Adiciona uma camada de classificação
    model.add(Dropout(0.2)) # Adiciona uma camada de dropout
    model.add(Dense(units=512, activation=den_activ)) # Adiciona uma camada de classificação
    model.add(Dropout(0.2)) # Adiciona uma camada de dropout
    model.add(Dense(units=256, activation=den_activ)) # Adiciona uma camada de classificação
    model.add(Dropout(0.2)) # Adiciona uma camada de dropout
    model.add(Dense(units=n_classes, activation=out_activ)) # Adiciona uma camada de saída

    # Compila o modelo
    model.compile(loss=loss,
            optimizer=optimizer,
            metrics=metrics)

    # Retorna o modelo e os parametros
    return model, str_hparameters