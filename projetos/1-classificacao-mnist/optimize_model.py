import tensorflow as tf
from tensorflow import keras
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------
MODELPATH = "model.h5"
COMPACTED_MODELPATH = "model.tflite"

def convert_model():
    """
    Converte o modelo Keras para TensorFlow Lite aplicando quantização dinâmica.
    A intenção aqui é encolher o bicho pra caber num dispositivo embarcado
    sem perder tanta precisão.
    """
    # Carrego o modelo original do disco — essa é a fera que vamos espremer
    model = keras.models.load_model(MODELPATH)
    
    # Crio o conversor que sabe como transformar layers do Keras em ops do TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Ligar a otimização DEFAULT faz o conversor usar Dynamic Range Quantization
    # Com isso os pesos caem de float32 pra int8 (só os pesos, as ativações seguem float)
    # É um meio-termo muito esperto entre tamanho e precisão
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Essa chamada da conversão propriamente dita 
    # Devolve os bytes do modelo comprimidos
    tflite_model = converter.convert()


    with open(COMPACTED_MODELPATH, "wb") as f:
        f.write(tflite_model)

    # Pego o tamanho dos dois arquivos e divido por 1024 pra enxergar em KB
    original_size = os.path.getsize(MODELPATH) / 1024
    optimized_size = os.path.getsize(COMPACTED_MODELPATH) / 1024

    # A continha da redução percentual — quanto menor o número, mais feliz eu fico
    red = (1 - optimized_size / original_size) * 100

    # E agora imprime os números pra ver se valeu a pena
    print(f"Tamanho antigo (model.h5): {original_size:.2f} KB")
    print(f"Tamanho otimizado (model.tflite): {optimized_size:.2f} KB")
    print(f"Redução de tamanho: {red:.2f}%")

if __name__ == "__main__":
    convert_model()