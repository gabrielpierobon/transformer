import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Masking

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
    Returns a mask of shape (batch_size, seq_len) where 1 indicates padding
    """
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return mask  # Shape: (batch_size, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate scaled dot product attention."""
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)

    return output, attention_weights

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

        self.wq = tf.keras.layers.Dense(d_model, name='query_projection')
        self.wk = tf.keras.layers.Dense(d_model, name='key_projection')
        self.wv = tf.keras.layers.Dense(d_model, name='value_projection')
        self.dense = tf.keras.layers.Dense(d_model, name='output_projection')

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v, mask):
        batch_size = tf.shape(q)[0]

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        if mask is not None:
            mask = tf.expand_dims(mask[:, tf.newaxis, :], axis=1)
            mask = tf.broadcast_to(mask, [batch_size, self.num_heads, tf.shape(q)[2], tf.shape(k)[2]])

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)
        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    """
    Position-wise feed-forward network.
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu', name='ffn_first_layer'),
        tf.keras.layers.Dense(d_model, name='ffn_output_layer')
    ])

class CustomMSEWithUncertainty(tf.keras.losses.Loss):
    """Custom MSE loss that also outputs prediction uncertainty."""
    
    def call(self, y_true, y_pred):
        # Split the prediction into mean and log variance
        mean = y_pred[:, 0:1]
        log_var = y_pred[:, 1:2]
        
        # Calculate the loss
        precision = tf.exp(-log_var)
        mse = tf.square(y_true - mean)
        loss = 0.5 * precision * mse + 0.5 * log_var
        
        return tf.reduce_mean(loss)

def gaussian_nll(y_true, y_pred):
    """Gaussian Negative Log Likelihood Loss using pure TensorFlow operations"""
    mu, log_sigma = tf.split(y_pred, 2, axis=-1)
    sigma = tf.math.exp(0.5 * log_sigma)
    
    # Calculate negative log likelihood
    log_2pi = tf.math.log(2.0 * np.pi)
    log_likelihood = -0.5 * (
        log_2pi + 
        2.0 * log_sigma + 
        tf.square(y_true - mu) / tf.square(sigma)
    )
    return -tf.reduce_mean(log_likelihood)

def smape_loss(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (sMAPE) loss"""
    epsilon = 0.1  # small constant to avoid division by zero
    true_o = tf.cast(y_true, tf.float32)
    pred_o = tf.cast(y_pred, tf.float32)
    numerator = tf.abs(pred_o - true_o)
    denominator = tf.abs(true_o) + tf.abs(pred_o) + epsilon
    return 2 * tf.reduce_mean(numerator / denominator)

def hybrid_loss(alpha=0.9):
    """
    Create a hybrid loss function combining sMAPE and Gaussian NLL.
    
    Args:
        alpha (float): Weight for sMAPE loss. (1-alpha) will be used for Gaussian NLL.
        
    Returns:
        function: Hybrid loss function
    """
    def loss_fn(y_true, y_pred):
        # Split predictions into mu and sigma
        mu, log_sigma = tf.split(y_pred, 2, axis=-1)
        
        # Compute sMAPE Loss using mu as the point forecast
        smape_value = smape_loss(y_true, mu)
        
        # Compute Gaussian NLL
        nll_value = gaussian_nll(y_true, y_pred)
        
        # Combined weighted loss
        return alpha * smape_value + (1.0 - alpha) * nll_value
    
    return loss_fn

def build_transformer_model(sequence_length=60, num_heads=4, d_model=512, dff=512, rate=0.05, probabilistic=True):
    """
    Build a transformer model for time series forecasting with masking and optional uncertainty estimation.
    """
    inputs = tf.keras.Input(shape=(sequence_length, 1), name='input_sequence')

    # Add masking layer
    masking = Masking(mask_value=0.0)(inputs)

    # Create padding mask for attention
    padding_mask = create_padding_mask(tf.squeeze(masking, axis=-1))

    # Project input to d_model dimensions
    x = tf.keras.layers.Dense(d_model)(masking)

    # Add positional encoding
    pos_encoding = positional_encoding(sequence_length, d_model)
    x = x + pos_encoding[:, :tf.shape(x)[1], :]

    # Multi-head attention
    attn_output, _ = CustomMultiHeadAttention(d_model, num_heads)(x, x, x, padding_mask)
    attn_output = tf.keras.layers.Dropout(rate, name='attention_dropout')(attn_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='attention_output_norm')(x + attn_output)

    # Feed forward network
    ffn_output = point_wise_feed_forward_network(d_model, dff)(out1)
    ffn_output = tf.keras.layers.Dropout(rate, name='ffn_output_dropout')(ffn_output)
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, name='ffn_output_norm')(out1 + ffn_output)

    # Global average pooling
    global_avg_pooling = tf.keras.layers.GlobalAveragePooling1D()(out2)

    if probabilistic:
        # Output mean and log variance for uncertainty estimation
        mean = tf.keras.layers.Dense(1, name='mean')(global_avg_pooling)
        log_var = tf.keras.layers.Dense(1, name='log_var')(global_avg_pooling)
        outputs = tf.keras.layers.Concatenate(name='probabilistic_output')([mean, log_var])
    else:
        # Point prediction output
        outputs = tf.keras.layers.Dense(1, name='output_layer')(global_avg_pooling)

    return tf.keras.Model(inputs=inputs, outputs=outputs)

def get_model(sequence_length=60, probabilistic=True, loss_type='gaussian_nll', loss_alpha=0.9):
    """
    Get the compiled transformer model.
    
    Args:
        sequence_length (int): Length of input sequences
        probabilistic (bool): Whether to use probabilistic predictions
        loss_type (str): Type of loss function ('gaussian_nll', 'smape', or 'hybrid')
        loss_alpha (float): Weight for sMAPE in hybrid loss (1-alpha for Gaussian NLL)
    """
    model = build_transformer_model(
        sequence_length=sequence_length,
        num_heads=4,
        d_model=512,
        dff=512,
        rate=0.05,
        probabilistic=probabilistic
    )
    
    if probabilistic:
        if loss_type == 'gaussian_nll':
            loss = gaussian_nll
        elif loss_type == 'smape':
            # For SMAPE with probabilistic model, we'll only use the mean prediction
            def smape_prob(y_true, y_pred):
                mu, _ = tf.split(y_pred, 2, axis=-1)
                return smape_loss(y_true, mu)
            loss = smape_prob
        elif loss_type == 'hybrid':
            loss = hybrid_loss(alpha=loss_alpha)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    else:
        loss = 'mse'
    
    # Custom MAE metric for probabilistic model that only uses mean prediction
    if probabilistic:
        def mae_prob(y_true, y_pred):
            mu, _ = tf.split(y_pred, 2, axis=-1)
            return tf.keras.metrics.mean_absolute_error(y_true, mu)
        metrics = [mae_prob]
    else:
        metrics = ['mae']
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=loss,
        metrics=metrics
    )
    
    return model 