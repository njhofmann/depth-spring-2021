def cuda_tensor_to_np_arr(tensor):
    return tensor.cpu().numpy()[0]