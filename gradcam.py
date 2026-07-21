import tensorflow as tf
import numpy as np
import cv2

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(

        inputs=model.inputs,

        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        loss = predictions[:,0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[...,tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap,0)

    heatmap /= tf.math.reduce_max(heatmap)+1e-10

    return heatmap.numpy()


def overlay_heatmap(img, heatmap, alpha=0.4):

    heatmap = cv2.resize(

        heatmap,

        (img.shape[1],img.shape[0])

    )

    heatmap = np.uint8(255*heatmap)

    heatmap = cv2.applyColorMap(

        heatmap,

        cv2.COLORMAP_JET

    )

    overlay = cv2.addWeighted(

        img,

        1-alpha,

        heatmap,

        alpha,

        0

    )

    return heatmap, overlay
