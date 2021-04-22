'''
A simplified 3-D Tensor (channels, height, weight) for convolutional neural networks.
'''


class Tensor(object):
    def __init__(self, c, h, w):
        self.c = c
        self.h = h
        self.w = w

    def __repr__(self) -> str:
        return f'({self.c}, {self.h}, {self.w})'

    def equals(self, other):
        return self.c == other.c and self.h == other.h and self.w == other.w

    def broadcastable(self, other):
        return (self.c % other.c == 0 or other.c % self.c == 0) and \
               (self.h % other.h == 0 or other.h % self.h == 0) and \
               (self.w % other.w == 0 or other.w % self.w == 0)


'''
Calculate the single-sample inference-time params and FLOPs of a convolutional
neural network with PyTorch-like APIs.

To calculate the params and FLOPs of certain network architecture, CNNCalculator
needs to be inherited and the network needs to be defined as in PyTorch.

For convenience, some basic operators are pre-defined and other modules can be
defined in a similar way. Parameters and FLOPs in Batch Normalization and other
types of layers are also computed. If only Convolutional and Linear layers are
considered, please modify the code.

Refer to `MobileNet.py` for details.
'''


class CNNCalculator(object):
    def __init__(self, only_mac=False):
        self.params = 0
        self.flops = 0
        self.only_mac = only_mac

    def calculate(self, *inputs):
        raise NotImplementedError

    def Conv2d(self, tensor, out_c, size, stride=1, padding=0, groups=1, bias=True, name='conv'):
        if type(size) == int:
            size = (size, size)
        if type(stride) == int:
            stride = (stride, stride)
        if type(padding) == int:
            padding = (padding, padding)
        assert type(size) == tuple and len(size) == 2, 'illegal size parameters'
        assert type(stride) == tuple and len(stride) == 2, 'illegal stride parameters'
        assert type(padding) == tuple and len(padding) == 2, 'illegal padding parameters'
        size_h, size_w = size
        stride_h, stride_w = stride
        padding_h, padding_w = padding

        in_c = tensor.c
        out_h = (tensor.h - size_h + 2 * padding_h) // stride_h + 1
        out_w = (tensor.w - size_w + 2 * padding_w) // stride_w + 1
        assert in_c % groups == 0 and out_c % groups == 0, 'in_c and out_c must be divisible by groups'

        self.params += out_c * in_c // groups * size_h * size_w
        self.flops += out_c * out_h * out_w * in_c // groups * size_h * size_w
        if bias:
            self.params += out_c
            self.flops += out_c * out_w * out_h

        return Tensor(out_c, out_h, out_w)

    def BatchNorm2d(self, tensor, name='batch_norm'):
        return tensor
        # Batch normalization can be combined with the preceding convolution, so there are no FLOPs.
        # out_c = tensor.c
        # out_h = tensor.h
        # out_w = tensor.w

        # if self.only_mac:
        # self.params += 4 * out_c
        # self.flops += out_c * out_h * out_w
        # return Tensor(out_c, out_h, out_w)

    def ReLU(self, tensor, name='relu'):
        out_c = tensor.c
        out_h = tensor.h
        out_w = tensor.w

        if not self.only_mac:
            self.flops += out_c * out_h * out_w
        return Tensor(out_c, out_h, out_w)

    def Sigmoid(self, tensor, name='relu'):
        out_c = tensor.c
        out_h = tensor.h
        out_w = tensor.w

        if not self.only_mac:
            self.flops += out_c * out_h * out_w
        return Tensor(out_c, out_h, out_w)

    def Pool2d(self, tensor, size, stride=1, padding=0, name='pool'):
        if type(size) == int:
            size = (size, size)
        if type(stride) == int:
            stride = (stride, stride)
        if type(padding) == int:
            padding = (padding, padding)
        assert type(size) == tuple and len(size) == 2, 'illegal size parameters'
        assert type(stride) == tuple and len(stride) == 2, 'illegal stride parameters'
        assert type(padding) == tuple and len(padding) == 2, 'illegal padding parameters'
        size_h, size_w = size
        stride_h, stride_w = stride
        padding_h, padding_w = padding

        out_c = tensor.c
        out_h = (tensor.h - size_h + 2 * padding_h) // stride_h + 1
        out_w = (tensor.w - size_w + 2 * padding_w) // stride_w + 1
        if not self.only_mac:
            self.flops += out_c * out_h * out_w * size_h * size_w
        return Tensor(out_c, out_h, out_w)

    def AvgPool2d(self, tensor, size, stride=1, padding=0, name='avg_pool'):
        return self.Pool2d(tensor, size, stride=stride, padding=padding, name=name)

    def MaxPool2d(self, tensor, size, stride=1, padding=0, name='max_pool'):
        return self.Pool2d(tensor, size, stride=stride, padding=padding, name=name)

    def GlobalAvgPool2d(self, tensor, name='global_avg_pool'):
        size = (tensor.h, tensor.w)
        return self.AvgPool2d(tensor, size)

    def GlobalMaxPool2d(self, tensor, name='global_max_pool'):
        size = (tensor.h, tensor.w)
        return self.MaxPool2d(tensor, size)

    def Linear(self, tensor, out_c, name='fully_connected'):
        in_c = tensor.c
        out_h = tensor.h
        out_w = tensor.w
        assert out_h == 1 and out_w == 1, 'out_h or out_w is greater than 1 in Linear layer.'
        self.params += in_c * out_c
        self.flops += in_c * out_c
        return Tensor(out_c, out_h, out_w)

    def Concat(self, tensors, name='concat'):
        out_c = 0
        out_h = tensors[0].h
        out_w = tensors[0].w
        for tensor in tensors:
            assert tensor.h == out_h and tensor.w == out_w, 'tensor dimensions mismatch in Concat layer.'
            out_c += tensor.c
        return Tensor(out_c, out_h, out_w)

    def MultiAdd(self, tensor, other, name='multi_add'):
        assert tensor.broadcastable(other), 'tensor dimensions mismatch in Add layer.'
        out_c = tensor.c
        out_h = tensor.h
        out_w = tensor.w
        if not self.only_mac:
            self.flops += out_c * out_h * out_w
        return Tensor(out_c, out_h, out_w)

    def Add(self, tensor, other, name='add'):
        return self.MultiAdd(tensor, other, name=name)

    def Multi(self, tensor, other, name='multi'):
        return self.MultiAdd(tensor, other, name=name)

    def SplitBySize(self, tensor, sizes, name='split_by_size'):
        assert sum(sizes) == tensor.c, 'sizes and tensor.c do not match.'
        return [Tensor(c, tensor.h, tensor.w) for c in sizes]


if __name__ == '__main__':
    t = Tensor(4, 425, 560)
    c = CNNCalculator()
    print(t)
    for _ in range(5):
        t = c.AvgPool2d(t, 3, padding=1, stride=2)
        print(t)