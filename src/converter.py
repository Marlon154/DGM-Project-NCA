import torch
import torch.nn as nn
from nca import NCA
import numpy as np
import base64
import json


def load_model(model_path):
    model = NCA()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def pad_to_multiple_of_4(arr):
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, (4 - arr.shape[-1] % 4) % 4)]
    return np.pad(arr, pad_width, mode='constant')


def pack_layer(weight, bias, outputType=np.uint8):
    if len(weight.shape) > 2:
        weight = weight.reshape(weight.shape[0], -1)

    in_ch, out_ch = weight.shape

    # Reshape bias to match weight's output dimension
    bias = np.repeat(bias, out_ch // len(bias))

    # Pad weight and bias to multiples of 4
    weight = pad_to_multiple_of_4(weight)
    bias = pad_to_multiple_of_4(bias)

    weight_scale, bias_scale = 1.0, 1.0
    if outputType == np.uint8:
        weight_scale = max(2.0 * np.abs(weight).max(), 1e-8)
        bias_scale = max(2.0 * np.abs(bias).max(), 1e-8)
        weight = np.round((weight / weight_scale + 0.5) * 255).clip(0, 255)
        bias = np.round((bias / bias_scale + 0.5) * 255).clip(0, 255)

    packed = np.vstack([weight, bias[None, ...]])
    packed = packed.reshape(in_ch + 1, -1, 4)
    packed = outputType(packed)
    packed_b64 = base64.b64encode(packed.tobytes()).decode('ascii')

    return {
        'data_b64': packed_b64,
        'in_ch': in_ch,
        'out_ch': out_ch,
        'weight_scale': float(weight_scale),
        'bias_scale': float(bias_scale),
        'type': outputType.__name__
    }


def export_ca_to_webgl_demo(ca, outputType=np.uint8):
    chn = ca.n_channels

    # Get the weights and biases from the first and last convolutions
    w1 = ca.conv[0].weight.data.cpu().numpy()
    b1 = ca.conv[0].bias.data.cpu().numpy() if ca.conv[0].bias is not None else np.zeros(ca.conv[0].out_channels)

    w2 = ca.conv[2].weight.data.cpu().numpy()
    b2 = ca.conv[2].bias.data.cpu().numpy() if ca.conv[2].bias is not None else np.zeros(ca.conv[2].out_channels)

    # Reorder the first layer inputs to meet webgl demo perception layout
    w1 = w1.reshape(chn, 3, -1).transpose(1, 0, 2).reshape(3 * chn, -1)

    layers = [
        pack_layer(w1, b1, outputType),
        pack_layer(w2.squeeze(), b2, outputType)  # Squeeze w2 to remove singleton dimensions
    ]
    return json.dumps(layers)


if __name__ == "__main__":
    model_path = "nca.pth"

    # Load the PyTorch model
    loaded_model = load_model(model_path)

    # Export the model for WebGL demo
    webgl_json = export_ca_to_webgl_demo(loaded_model)

    # Save the JSON to a file
    with open("nca_webgl.json", "w") as f:
        f.write(webgl_json)

    print("Model exported for WebGL demo and saved as nca_webgl.json")
