import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, BatchNormalization, GlobalAveragePooling3D, Dense, Multiply
from Evaluation import evaluation


def efficient_attention_3d(input_tensor):
    filters = input_tensor.shape[-1]  # Dynamically get the number of filters
    se = GlobalAveragePooling3D(keepdims=True)(input_tensor)  # Keep dimensions
    se = Dense(filters // 4, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    return Multiply()([input_tensor, se])  # Element-wise multiplication


def Model_3D_EfficientNet(Train_Data, Train_Target, Test_Data, Test_Target, batch_size, sol=None):

    if sol is None:
        sol = [5, 5, 1]
    Activations = ['linear', 'tanh', 'relu', 'sigmoid', 'softmax']

    IMG_SIZE = 16
    num_classes = Train_Target.shape[1]

    # Reshape data to fit 3D input
    Train_x = np.zeros((Train_Data.shape[0], IMG_SIZE, IMG_SIZE, IMG_SIZE, 1))
    for i in range(Train_Data.shape[0]):
        temp = np.resize(Train_Data[i], (IMG_SIZE * IMG_SIZE * IMG_SIZE, 1))
        Train_x[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, IMG_SIZE, 1))

    Test_X = np.zeros((Test_Data.shape[0], IMG_SIZE, IMG_SIZE, IMG_SIZE, 1))
    for i in range(Test_Data.shape[0]):
        temp = np.resize(Test_Data[i], (IMG_SIZE * IMG_SIZE * IMG_SIZE, 1))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, IMG_SIZE, 1))

    input_shape = (IMG_SIZE, IMG_SIZE, IMG_SIZE, 1)  # Assuming grayscale 3D images
    inputs = Input(shape=input_shape)

    # 3D Convolutional Layers
    x = Conv3D(int(sol[1]), kernel_size=(3, 3, 3), padding='same', activation=Activations[int(sol[2]) - 1])(inputs)
    x = BatchNormalization()(x)
    x = Conv3D(int(sol[1]), kernel_size=(3, 3, 3), padding='same', activation=Activations[int(sol[2]) - 1])(x)
    x = BatchNormalization()(x)

    # Efficient Attention Module
    x = efficient_attention_3d(x)

    # More CNN Layers
    x = Conv3D(int(sol[1]), kernel_size=(3, 3, 3), padding='same', activation=Activations[int(sol[2]) - 1])(x)
    x = BatchNormalization()(x)
    x = efficient_attention_3d(x)

    # Global Pooling & Fully Connected Layer
    x = GlobalAveragePooling3D()(x)
    x = Dense(num_classes, activation=Activations[int(sol[2]) - 1])(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # Train model
    model.fit(Train_x, Train_Target, epochs=int(sol[1]), batch_size=batch_size, validation_data=(Test_X, Test_Target))

    # Predictions & Evaluation
    pred = model.predict(Test_X)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, Test_Target)
    return Eval, pred

