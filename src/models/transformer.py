import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

def positional_encoding(position, d_model):
    """
    Create positional encoding matrix for transformer input.
    """
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
    """
    Create a mask for padding tokens (zeros).
    """
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]

class CustomMultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention layer with proper masking support.
    """
    def __init__(self, d_model, num_heads):
        super(CustomMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        # Scaled dot-product attention with masking
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(output)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    """
    Position-wise feed-forward network.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

def build_transformer_model(num_heads=4, d_model=512, dff=2048, dropout_rate=0.1):
    """
    Build a transformer model for time series forecasting with masking and probabilistic output.
    """
    inputs = tf.keras.Input(shape=(60, 1))  # 60 time steps, 1 feature

    # Add masking layer
    masked = tf.keras.layers.Masking(mask_value=0.0)(inputs)

    # Create padding mask for attention
    padding_mask = create_padding_mask(tf.squeeze(masked, axis=-1))

    # Add positional encoding
    pos_encoding = positional_encoding(60, d_model)
    x = tf.keras.layers.Dense(d_model)(masked)  # Project to d_model dimensions
    x = x + pos_encoding[:, :tf.shape(x)[1], :]
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # Transformer blocks
    for _ in range(2):  # 2 transformer blocks
        # Multi-head attention
        attn_output, _ = CustomMultiHeadAttention(d_model, num_heads)(x, x, x, padding_mask)
        attn_output = tf.keras.layers.Dropout(dropout_rate)(attn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed forward
        ffn_output = point_wise_feed_forward_network(d_model, dff)(x)
        ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    # Global average pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Probabilistic output
    mu = tf.keras.layers.Dense(1, name='mu')(x)  # Mean
    sigma = tf.keras.layers.Dense(1, activation='softplus', name='sigma')(x)  # Standard deviation
    outputs = tf.keras.layers.Concatenate(name='distribution_params')([mu, sigma])

    return tf.keras.Model(inputs=inputs, outputs=outputs)

def gaussian_nll(y_true, y_pred):
    """
    Gaussian negative log likelihood loss function.
    """
    mu, sigma = tf.split(y_pred, 2, axis=-1)
    dist = tfp.distributions.Normal(loc=mu, scale=sigma)
    return -tf.reduce_mean(dist.log_prob(y_true))

def get_model(sequence_length=60):
    """
    Get the compiled transformer model.
    """
    model = build_transformer_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=gaussian_nll,
        metrics=['mae']
    )
    return model 