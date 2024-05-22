from model import createModel
import os, glob, cv2, tf2onnx, onnx
import numpy as np
import tensorflow as tf

neg_images = glob.glob("trainDataNegative/*.jpg")
pos_images = glob.glob("trainDataPositive/*.jpg")
row_neg_images = [img for img in neg_images if 'row' in img]
col_neg_images = [img for img in neg_images if 'col' in img]
row_pos_images = [img for img in pos_images if 'row' in img]
col_pos_images = [img for img in pos_images if 'col' in img]

def save_model_to_onnx(model, output_name):
    model.output_names = ['output']
    input_signature = [
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name='row'),
        tf.TensorSpec(model.inputs[1].shape, model.inputs[1].dtype, name='col'),
        ]
    onnx_model, _ = tf2onnx.convert.from_keras(model,input_signature)
    onnx.save(onnx_model, output_name)

X1, X2, y = [], [], []
shape = (256, 256)
for i, neg_img in enumerate(row_neg_images):
  X1.append(cv2.imread(row_neg_images[i])/255.0)
  X2.append(cv2.imread(col_neg_images[i])/255.0)
  y.append([1,0])
for i, pos_img in enumerate(row_pos_images):
  X1.append(cv2.imread(row_pos_images[i])/255.0)
  X2.append(cv2.imread(col_pos_images[i])/255.0)
  y.append([0,1])
X1, X2, y = np.array(X1), np.array(X2), np.array(y)

fake_model = createModel(shape[0], shape[1], 3)
fake_model.compile(optimizer='adam', loss = 'categorical_crossentropy',metrics=['accuracy'])
fake_model.fit([X1, X2], y, batch_size=10, epochs=100, verbose=1)
save_model_to_onnx(fake_model, 'fake_detect.onnx')