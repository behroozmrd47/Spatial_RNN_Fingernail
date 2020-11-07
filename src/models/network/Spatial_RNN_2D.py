"""
@author: Behrooz Bajestani (behrooz.mrd47@gmail.com)
"""
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from collections import OrderedDict
from tensorflow.python.keras import initializers
from tensorflow.python.keras import activations
import warnings


class SpatialRNN2D(tf.keras.layers.Layer):
    def __init__(self, rnn_seq_length, activation='relu', kernel_initializer='random_uniform', merge_mode='concat',
                 output_conv_filter=None, **kwargs):
        """
        Class for Spatial RNN layer capable of learning spatial connections between pixels of an 2D image in an RNN
        fashion along all four directions of up, downs,left and right. Implemented in tensorflow 2.0 with Keras API.
        The RNN unit is plain RNN with ReLu activation function (default) as suggested by Li et. al. (2019).
        The activation function can be chosen from activation function available from tensorflow.python.keras library.
        The spatial connections will be analysed in all principal directions sweeping to right, left, down, up.
        The results from spatial RNN analysis in each of principal directions would have exact same shape of input
        and can be concatenated or merged together depending on "merge_mode" input parameter.
        The "merge_mode" input parameter can be set to either 'concat' (default) or 'convolution'.
        By default, the results for all principal directions will be concatenated together resulting in a final output
        shape with 4 times number of channels as the input channels. In case of 'convolution' merge mode, the results
        for all principal directions will be concatenated and then the number of channels will be converted using a
        1*1 2D convolution layer. The output number of channels will be determined based on the 'output_conv_filter'
        input parameter which by default is et to input shape number of channels.

        The input 2D image is recommended to be square as sufficient testing with non-square input images has not been
        done. When using this layer as the first layer, preceded it with an Keras "Input" layer. Should be used with
        `data_format="channels_last"`. The kernel initializer and activation functions can be set using the ones
        available in tensorflow.python.keras.initializers & tensorflow.python.keras.activations.

        Examples:
        The inputs are 5x5 RGB images with `channels_last` and the batch of 1
        input_shape = (2, 5, 5, 3)  # (batch, height, width, channels)
        x_in = tf.keras.layers.Input((5, 5, 3))
        spatial_rnn_concat = SpatialRNN2D(rnn_seq_length=4, merge_mode='concat')
        spatial_rnn_merge = SpatialRNN2D(rnn_seq_length=4m, merge_mode='convolution')
        y_out = spatial_rnn_concat(x_in)  # output shape of (2, 5, 5, 12)
        y_out = spatial_rnn_merge(x_in)  # output shape of (2, 5, 5, 3)

        :param rnn_seq_length: Integer, the length of pixels sequence to be analysed by RNN unit
        :param activation: (relu) Activation function used after following Spatial RNN and merge convolution layers.
        :param kernel_initializer: (random_uniform) Initializer for the `kernel` weights matrix.
        :param merge_mode: ('concat') To concatenate or merge (by 'convolution') the result for each direction pass.
        :param output_conv_filter: number of output channels in case 'convolution' merge mode is selected.
        """

        super().__init__(**kwargs)
        self.padding = "same"
        if merge_mode not in ['concat', 'convolution']:
            raise ValueError('Unknown merge mode: the merge mode argument can be either \'concat\' or \'convolution\'.')
        self.merge_mode = merge_mode
        self.output_conv_filter = output_conv_filter
        self.seq_length = rnn_seq_length
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_dic = OrderedDict()
        self.kernel_switch_dic = OrderedDict({'right': np.array([[1, 0, 0]]),
                                              'left': np.array([[0, 0, 1]]),
                                              'down': np.array([[1], [0], [0]]),
                                              'up': np.array([[0], [0], [1]])})

    def build(self, input_shape):
        """
        Build the class based on the input shape and the direction parameter. The required RNN and convolution merge
        kernels are built as well.

        :param input_shape: 4D tensor with shape: `(batch_shape, rows, cols, channels)`.
        Raises:
            Warning if the rnn sequence length is greater than input image edge size.
        """
        self.in_channel = int(input_shape[-1])
        if self.merge_mode == 'convolution' and self.output_conv_filter is None:
            self.output_conv_filter = self.in_channel

        if self.seq_length > input_shape[-3] or self.seq_length > input_shape[-2]:
            warnings.warn("The rnn sequence length parameter is equal or bigger than image edge size. This will not "
                          "have any effect on the results but will increase computation cost. You can change the "
                          "rnn sequence length parameter to as big as (edge size - 1).")

        for direction, kernel_switch in self.kernel_switch_dic.items():
            self.kernel_switch_dic[direction] = self.get_kernel_switch(kernel_switch)

        for direction, kernel_switch in self.kernel_switch_dic.items():
            self.kernel_dic[direction] = self.add_weight(
                shape=kernel_switch.shape, initializer=self.kernel_initializer, trainable=True)
        if self.merge_mode == 'convolution':
            self.conv_kernel = self.add_weight(
                shape=(1, 1, self.in_channel * 4, self.output_conv_filter), initializer=self.kernel_initializer,
                trainable=True)
        super().build(input_shape)

    @tf.function
    def call(self, input_tensor):
        """
        Calls the tensor for forward pass operation.

        :param input_tensor: The input dataset of 2D images with shape of `(batch_shape, rows, cols, channels)`.
        :return: 4D tensor with shape: `(batch_shape, rows, cols, input_image_channels * 4)` for 'concat' merge mode.
        4D tensor with shape: `(batch_shape, rows, cols, output_conv_filter)` for 'convolution' merge mode.
        """
        input_tensor = K.cast(tf.identity(input_tensor), tf.float32)
        result_tensors_list_img = []
        for direction, kernel in self.kernel_dic.items():
            res_sum = tf.identity(input_tensor)
            tensor = tf.identity(input_tensor)
            for i in range(self.seq_length):
                conv = K.depthwise_conv2d(x=tensor, depthwise_kernel=kernel * self.kernel_switch_dic[direction],
                                          padding='same')
                tensor = self.activation(conv)
                res_sum += tensor
            result_tensors_list_img.append(res_sum)
        result_tensors_list_img = K.concatenate(result_tensors_list_img, axis=-1)
        if self.merge_mode == 'convolution':
            result_tensors_list_img = K.conv2d(x=result_tensors_list_img, kernel=self.conv_kernel, padding='same')
            result_tensors_list_img = self.activation(result_tensors_list_img)
        return result_tensors_list_img

    def compute_output_shape(self, input_shape):
        """
        Compute output shape.

        :param input_shape: 4D tensor with shape: `1 + (rows, cols, channels)`.
        :return: 4D tensor with shape: `(batch_shape, rows, cols, input_image_channels * 4)` for 'concat' merge mode.
                4D tensor with shape: `(batch_shape, rows, cols, output_conv_filter)` for 'convolution' merge mode.
        """

        if self.merge_mode == 'concat':
            return input_shape[0], input_shape[1], input_shape[2], self.in_channel * 4
        else:
            return input_shape[0], input_shape[1], input_shape[2], self.output_conv_filter

    def get_kernel_switch(self, kernel_switch):
        """
        Compute the ker nel switch.

        :param kernel_switch: The kernel switch in format of numpy array consisting of zeros and ones.
        :return: The tensor format of kernel switch consisting of zeros and ones. The kernel size would be
                (kernel_height, kernel_width, input_layer_channels, 1)
        """
        kernel_switch = np.repeat(kernel_switch[:, :, np.newaxis], int(self.in_channel), axis=-1)
        kernel_switch = np.expand_dims(kernel_switch, -1)
        return K.constant(kernel_switch, dtype=tf.float32)

    def get_config(self):
        """
        Overwrite get_config method of layer class.

        :return: config dictionary plus the custom class parameters initialized.
        """
        config = super().get_config().copy()
        config.update({
            'padding': self.padding,
            'seq_length': self.seq_length,
            'activation': self.activation,
            'kernel_initializer': self.kernel_initializer,
            'merge_mode': self.merge_mode,
            'output_conv_filter': self.output_conv_filter,
        })
        return config
