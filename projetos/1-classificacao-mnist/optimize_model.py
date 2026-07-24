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
    model = keras.models.load_model(MODELPATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # aplicação da técnica de otimização:
    # o modo DEFAULT ativa a Dynamic Range Quantization
    # para converter os pesos de float32 para int8
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # execução da conversão:
    # gera os bytes do modelo já otimizado;
    tflite_model = converter.convert()


    with open(COMPACTED_MODELPATH, "wb") as f:
        f.write(tflite_model)
    
    # obtenção do tamanho dos arquivos:
    # converte de bytes para KB para facilitar a leitura;
    original_size = os.path.getsize(MODELPATH) / 1024
    optimized_size = os.path.getsize(COMPACTED_MODELPATH) / 1024


    # cálculo da redução percentual:
    # indica o quanto o modelo diminuiu após a quantização;
    red = (1 - optimized_size / original_size) * 100
  
    print(f"Tamanho original (model.h5): {original_size:.2f} KB")
    print(f"Tamanho otimizado (model.tflite): {optimized_size:.2f} KB")
    print(f"Redução de tamanho: {red:.2f}%")


if __name__ == "__main__":
    convert_model()