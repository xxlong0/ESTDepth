import cv2
import numpy as np
import torch


def colorize_depth(input, max_depth, color_mode=cv2.COLORMAP_RAINBOW):
    input_tensor = input.detach().cpu().numpy()
    normalized = input_tensor / max_depth * 255.0
    normalized = normalized.astype(np.uint8)
    if len(input_tensor.shape) == 3:
        normalized_color = np.zeros((input_tensor.shape[0],
                                     input_tensor.shape[1],
                                     input_tensor.shape[2],
                                     3))
        for i in range(input_tensor.shape[0]):
            normalized_color[i] = cv2.applyColorMap(normalized[i], color_mode)
        return torch.from_numpy(normalized_color).permute(0, 3, 1, 2)
    if len(input_tensor.shape) == 2:
        normalized = cv2.applyColorMap(normalized, color_mode)
        return torch.from_numpy(normalized).permute(2, 0, 1)


def colorize_depth_np(input, max_depth, color_mode=cv2.COLORMAP_RAINBOW):
    input_tensor = input
    input_tensor[input_tensor > max_depth] = max_depth
    normalized = input_tensor / max_depth * 255.0
    normalized = normalized.astype(np.uint8)
    if len(input_tensor.shape) == 3:
        normalized_color = np.zeros((input_tensor.shape[0],
                                     input_tensor.shape[1],
                                     input_tensor.shape[2],
                                     3))
        for i in range(input_tensor.shape[0]):
            normalized_color[i] = cv2.applyColorMap(normalized[i], color_mode)
        return normalized_color
    if len(input_tensor.shape) == 2:
        normalized = cv2.applyColorMap(normalized, color_mode)
        return normalized


def colorize_probmap(input):
    input_tensor = input.detach().cpu().numpy()
    input_tensor = input_tensor * 255.0
    input_tensor = input_tensor.astype(np.uint8)
    if len(input_tensor.shape) == 3:
        normalized_color = np.zeros((input_tensor.shape[0],
                                     input_tensor.shape[1],
                                     input_tensor.shape[2],
                                     3))
        for i in range(input_tensor.shape[0]):
            normalized_color[i] = cv2.applyColorMap(input_tensor[i], cv2.COLORMAP_RAINBOW)
        return torch.from_numpy(normalized_color).permute(0, 3, 1, 2)
    if len(input_tensor.shape) == 2:
        normalized = cv2.applyColorMap(input_tensor, cv2.COLORMAP_RAINBOW)
        return torch.from_numpy(normalized).permute(2, 0, 1)


def colorize_normal(input):
    return (input / 2. + 0.5)
