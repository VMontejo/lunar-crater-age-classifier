import tensorflow as tf
import numpy as np
import cv2



def make_gradcam_heatmap(img_array, model, last_conv_layer_name, original_image, pred_index=None):
    """
    Generates a Grad-CAM heatmap for a given image and model.

    This iterates through the layers manually to capture
    intermediate outputs.

    Parameters:
    -----------
    img_array : numpy array
        The input image tensor (must include batch dimension, e.g., (1, 224, 224, 3)).
    model : tf.keras.Model
        The trained classification model.
    last_conv_layer_name : str
        The name of the last convolutional layer (e.g., 'conv2d_4' or 'resnet50').
    pred_index : int, optional
        The index of the class to visualize. If None, uses the model's top prediction.

    Returns:
    --------
    heatmap : numpy array
        A 2D heatmap (float32) of shape (H, W) with values between 0 and 1.
    """
    # 1. Find the index of your target Conv layer
    conv_layer_index = None
    for i, layer in enumerate(model.layers):
        if layer.name == last_conv_layer_name:
            conv_layer_index = i
            break

    if conv_layer_index is None:
        raise ValueError(f"Could not find layer named {last_conv_layer_name}")

    # 2. Run the image through the model manually
    with tf.GradientTape() as tape:

        x = tf.cast(img_array, tf.float32)

        # A. Run through layers UP TO the conv layer
        for layer in model.layers[:conv_layer_index + 1]:
            x = layer(x)

        # B. Capture the conv output
        conv_output = x
        tape.watch(conv_output)

        # C. Run through the REMAINING layers (Pooling, Dense, etc.)
        for layer in model.layers[conv_layer_index + 1:]:
            x = layer(x)

        preds = x

        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 3. Calculate Gradients
    grads = tape.gradient(class_channel, conv_output)

    # 4. Global Average Pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # 5. Multiply and create heatmap
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]


    # 6. Post-Processing
    # Apply ReLU (remove negative values)
    heatmap = tf.maximum(heatmap, 0)

    # Normalize between 0 and 1
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val
    else:
        # Handle edge case where heatmap is empty/zero
        heatmap = tf.zeros(heatmap.shape)

    heatmap_2d = tf.squeeze(heatmap).numpy()

    #Resize, colorize, Overlay
    h, w, _ = original_image.shape
    heatmap_resized = cv2.resize(heatmap_2d, (w, h))

    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_image.astype(np.uint8), 0.6, heatmap_colored, 0.4, 0)


    return overlay, heatmap_resized
