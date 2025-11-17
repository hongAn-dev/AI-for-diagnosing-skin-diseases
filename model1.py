import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def load_both_models():
    """Tải cả MobileNetV2 và ResNet50"""
    
    # Tải MobileNetV2
    mobilenet_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    mobilenet_model.trainable = False
    
    # Tải ResNet50
    resnet_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    resnet_model.trainable = False
    
    return mobilenet_model, resnet_model

# Sử dụng hàm
mobilenet, resnet = load_both_models()

print("✅ Đã tải thành công cả 2 mô hình!")