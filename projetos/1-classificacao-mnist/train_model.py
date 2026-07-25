import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

def prepare_dataset():

    # Carregamento do dataset MNIST via keras: retorna tensores uint8
    # Shapes (60000,28,28) treino e (10000,28,28) teste, labels 0-9 já separado.
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()


    # Normalização dos pixels 
    # Conversão de uint8  para float32 no intervalo [0,1] 
    # Afim de evitar overflow e acelerar o treino
    # Ajusta o formato das imagens, insere canal único (1) com o açucar sintatico [..., None]
    # Entregando tensores 4D (60000,28,28,1) compatíveis com a assinatura da Conv2D
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test  = (x_test.astype("float32") / 255.0)[..., None]


    return (x_train, y_train), (x_test, y_test)

def create_model():
# Inicializa o modelo sequencial
    net = keras.models.Sequential(name="mnist_classifier")

    # Define o formato de entrada (28x28 pixels, 1 canal de cor)
    net.add(keras.Input(shape=(28, 28, 1)))
    
    # Extração de Características
    
    # Bloco 01 convulacionbal 
    # Aprende características simples (Detecção de bordas e padrões simples)
    # Usando 32 filtros 3x3
    net.add(layers.Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))

    # Mantém esses números em uma escala controlada
    # Evitando falhas matemáticas durante o aprendizado
    net.add(layers.BatchNormalization())

    # Diminui a dimensão espacial preservando as características mais importantes
    net.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Bloco 02 convulacionbal (Padrões intermediários)
    # aprende padrões mais complexos combinando as características extraídas anteriormente
    net.add(layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    net.add(layers.BatchNormalization())
    net.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Bloco 03 Convolucional (Características de alto nível)
    # Importantes para a classificação final
    net.add(layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    net.add(layers.BatchNormalization())
    net.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Classificação
    
    # Transforma a saída convolucional em um vetor unidemnsional
    # Para fornerver para as camadas ocultas
    net.add(layers.Flatten())
    
    # Camada oculta
    # Combina todas as caracteristicas aprendidas
    net.add(layers.Dense(units=128, activation="relu"))
    
    # Dropout para regularização (reduzir overfitting)
    net.add(layers.Dropout(rate=0.3))
    
    # Camada de saída com 10 neurônios (probabilidade para os dígitos 0-9)
    net.add(layers.Dense(units=10, activation="softmax"))

    # Configuração de otimização e compilação
    adam_opt = keras.optimizers.Adam(learning_rate=1e-3)
    
    net.compile(
        optimizer=adam_opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Exibe a arquitetura final no console
    net.summary()

    return net

def train_model(model, X_train, Y_train):
    #Executa o ciclo de treinamento da rede neural com monitoramento de validação.

    # Previne overfitting interrompendo o treino se a rede parar de aprender
    es_callback = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True
    )

    # Hiperparâmetros do treinamento
    num_epochs = 10
    batch_size = 32
    val_ratio = 0.1

    # Inicia o treinamento reservando uma fração dos dados para validação
    fit_history = model.fit(
        X_train,
        Y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=val_ratio,
        callbacks=[es_callback],
        verbose=1
    )

    # Recupera os melhores resultados atingidos durante as épocas
    best_val_acc = max(fit_history.history['val_accuracy'])
    best_val_loss = min(fit_history.history['val_loss'])

    print("\n" + "-"*40)
    print(f"Métricas Finais de Validação:")
    print(f"-> Acurácia: {best_val_acc:.4f}")
    print(f"-> Loss: {best_val_loss:.4f}")
    print("-"*40 + "\n")

    return fit_history


def evaluate_model(model, X_test, Y_test):
    #Afere a capacidade de generalização do modelo em dados não vistos
    
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)

    print("Desempenho no Conjunto de Teste:")
    print(f"-> Acurácia: {accuracy:.4f}")
    print(f"-> Loss: {loss:.4f}\n")


def save_model(model, filename="model.h5"):
    #Exporta a arquitetura e os pesos aprendidos para o disco
    
    model.save(filename)
    print(f"Sucesso: Rede neural exportada para '{filename}'.")


def main():
    
    # 1. Carregamento e preparação dos dados
    (X_train, Y_train), (X_test, Y_test) = prepare_dataset()

    # 2. Definição da arquitetura
    classifier = create_model()

    # 3. Processo de otimização (treino)
    train_model(classifier, X_train, Y_train)

    # 4. Verificação final
    evaluate_model(classifier, X_test, Y_test)

    # 5. Persistência
    save_model(classifier)


if __name__ == "__main__":
    main()