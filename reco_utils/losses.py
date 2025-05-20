
#=== SparseFocalLoss serializable ===
@register_keras_serializable()
class SparseFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=None, class_weights=None, from_logits=False):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

        # Préparer les poids de classes
        self.class_weights = None
        if isinstance(class_weights, dict):
            max_index = max(class_weights.keys()) + 1
            weights_array = [class_weights.get(i, 1.0) for i in range(max_index)]
            self.class_weights = tf.constant(weights_array, dtype=tf.float32)
        elif class_weights is not None:
            self.class_weights = tf.constant(class_weights, dtype=tf.float32)

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred)

        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])

        probs = tf.reduce_sum(y_pred * y_true_one_hot, axis=-1)
        focal = tf.pow(1.0 - probs, self.gamma)

        if self.alpha is not None:
            focal *= self.alpha

        if self.class_weights is not None:
            weights = tf.gather(self.class_weights, y_true)
            focal *= weights

        loss = -focal * tf.math.log(tf.maximum(probs, tf.keras.backend.epsilon()))
        return tf.reduce_mean(loss)
