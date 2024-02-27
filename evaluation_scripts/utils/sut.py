import numpy as np
import tensorflow as tf

def save_sut_labels(labels, file):
    with open(file, mode='w') as fp:
        for label in labels:
            linear, angular = labels[label]
            fp.write(f'{label}:({linear}, {angular})\n')

def load_sut_labels(file):
    labels = {}
    with open(file, mode='r') as fp:
        for line in fp:
            # 26120231033-img0.jpg:(0.18, 0.09)
            split = line.split(':')
            if len(split) == 2:
                labels[split[0].strip()] = tuple(map(
                    lambda v: float(v.strip()),
                    split[1].strip()[1:-1].split(',')
                ))
    return labels

def get_sut_model(mcdropout=False, dropout_rate=.1):
    from keras import Model
    from keras.layers import Conv2D, Dense, Dropout, Flatten, Input
    dropout_args = { 'training': True } if mcdropout else {}
    img_in = Input(shape=(120, 160, 1), name='img_in')
    x = img_in
    x = Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Dropout(rate=dropout_rate)(x, **dropout_args)
    x = Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Dropout(rate=dropout_rate)(x, **dropout_args)
    x = Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Dropout(rate=dropout_rate)(x, **dropout_args)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
    x = Dropout(rate=dropout_rate)(x, **dropout_args)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
    x = Dropout(rate=dropout_rate)(x, **dropout_args)
    x = Flatten(name='flattened')(x)
    x = Dense(units=100, activation='linear')(x)
    x = Dropout(rate=dropout_rate)(x, **dropout_args)
    x = Dense(units=50, activation='linear')(x)
    x = Dropout(rate=dropout_rate)(x, **dropout_args)
    linear = Dense(units=1, activation='linear', name='linear')(x)
    angular = Dense(units=1, activation='linear', name='angular')(x)
    model = Model(inputs=[img_in], outputs=[linear, angular])
    return model

def keras2tflite(model):
    tflite_interpreter = tf.lite.Interpreter(
        model_content=tf.compat.v2.lite.TFLiteConverter.from_keras_model(model).convert()
    )
    tflite_interpreter.allocate_tensors()
    return tflite_interpreter

def load_sut_model(model_path):
    if model_path.endswith('.tflite'):
        tflite_interpreter = tf.lite.Interpreter(model_path=model_path)
        tflite_interpreter.allocate_tensors()
        return tflite_interpreter
    else:
        return tf.keras.models.load_model(model_path)
    
def execute_sut_model(model, image):
    if issubclass(type(model), tf.lite.Interpreter):
        return execute_sut_model_tflite(model, image)
    else:
        return execute_sut_model_keras(model, image)

def execute_sut_model_tflite(tflite_interpreter, image):
    if issubclass(type(image), tf.keras.utils.Sequence): # TODO: I think this is not needed, remove
        results = []
        for batch in image:
            for img in batch:
                results.append(execute_sut_model(tflite_interpreter, image=img[0].astype(np.float32)))
        return results
    else:
        input_details = tflite_interpreter.get_input_details()
        output_details = tflite_interpreter.get_output_details()
        tflite_interpreter.set_tensor(input_details[0]["index"], [image])
        tflite_interpreter.invoke()
        if len(output_details) > 1:
            # LeoRover
            linear_x = tflite_interpreter.get_tensor(output_details[0]["index"])[0][0]
            angular_z = tflite_interpreter.get_tensor(output_details[1]["index"])[0][0]
            return angular_z # TODO: Consider all outputs, not just the angle?
        else:
            # Dave2
            return tflite_interpreter.get_tensor(output_details[0]["index"])[0][0]

def execute_sut_model_keras(model, image):
    if issubclass(type(image), tf.keras.utils.Sequence): # TODO: I think this is not needed, remove
        results = []
        for batch in image:
            for img in batch:
                results.append(execute_sut_model_keras(model, image=img[0].astype(np.float32)))
        return results
    else:
        if image.shape[0] not in [None, -1]:
            image = image.reshape(-1, *image.shape) # Tensorflow nonsense to identify sequences VS individual inputs
        #return float(model.predict(image, batch_size=1))
        result = model.predict(image, verbose=0)
        if max(result.shape) <= 1:
            return float(result)
        else:
            return result[1][0][0] # Steering angle
