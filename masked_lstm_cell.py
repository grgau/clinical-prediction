import tensorflow as tf
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.utils import tf_utils
from tensorflow.keras import activations

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class MaskedLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
  def __init__(self,
               num_units,
               forget_bias=1.0,
               state_is_tuple=True,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs):

    # Inputs must be 2-dimensional.
    self.input_spec = input_spec.InputSpec(ndim=2)

    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = tf.tanh

  @property
  def state_size(self):
    return (tf.nn.rnn_cell.LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError(
          "Expected inputs.shape[-1] to be known, "
          f"received shape: {inputs_shape}")

    input_depth = inputs_shape[-1]
    h_depth = self._num_units
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + h_depth, 4 * self._num_units])
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[4 * self._num_units],
        initializer=tf.compat.v1.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    sigmoid = tf.sigmoid
    one = tf.constant(1, dtype=tf.int32)

    if self._state_is_tuple:
      c, h = state
    else:
      c, h = tf.split(value=state, num_or_size_splits=2, axis=one)

    gate_inputs = tf.matmul(
        tf.concat([inputs, h], 1), self._kernel)
    gate_inputs = tf.nn.bias_add(gate_inputs, self._bias)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = tf.split(
        value=gate_inputs, num_or_size_splits=4, axis=one)

    forget_bias_tensor = tf.constant(self._forget_bias, dtype=f.dtype)

    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    add = tf.add
    multiply = tf.multiply
    new_c = add(
        multiply(c, sigmoid(add(f, forget_bias_tensor))),
        multiply(sigmoid(i), self._activation(j)))
    new_h = multiply(self._activation(new_c), sigmoid(o))

    if self._state_is_tuple:
      new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
    else:
      new_state = tf.concat([new_c, new_h], 1)
    return new_h, new_state

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "forget_bias": self._forget_bias,
        "state_is_tuple": self._state_is_tuple,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(MaskedLSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
