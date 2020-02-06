import numpy as np

np.random.seed(0)


class LinearClassifier(object):
    def __init__(self, num_classes, dimensions):
        self.num_classes = num_classes

        self.W = 0.01 * np.random.randn(dimensions, num_classes)
        self.b = np.zeros((1, num_classes))

    def _softmax(self, logits):
        logits_exp = np.exp(logits)
        logits_sum = np.sum(logits_exp, axis=1, dtype=float)
        probs = logits_exp / logits_sum

        return probs

    def _cross_entropy(self, predictions, targets):
        num_examples = predictions.shape[0]
        correct_log_probs = -np.log(predictions[range(num_examples), targets])
        crossentropy = np.sum(correct_log_probs) / num_examples

        return crossentropy

    def _l2_loss(self, params):
        reg = 0.0

        for param in params:
            reg += 0.5 * np.sum(param * param) * self.reg_lambda

        return reg

    def _derivaive_loss_logits(self, predictions, targets):
        num_examples = predictions.shape[0]
        dlogits = predictions[range(self.num_classes), targets] - 1
        dlogits /= num_examples
        return dlogits

    def _derivative_loss_W(self, X, dlogits):
        dW = np.dot(X.T, dlogits)
        return dW

    def _derivative_loss_b(self, dlogits):
        db = np.sum(dlogits, axis=0, keepdims=True)
        return db

    def predictions(self, X):
        logits = np.matmul(X, self.W) + self.b
        probs = self._softmax(logits)

        return probs

    def compute_loss(self, predictions, targets):
        loss = self._cross_entropy(predictions, targets)
        reg = self._l2_loss(self.w)
        return loss + reg

    def update(self):
        targets = y
        dW = self._derivative_loss_W(self._derivaive_loss_logits(self.predictions(X), targets))
        db = self._derivative_loss_b()

        self.W += -self.w + self.learning_rate * dW
        self.b += -self.b + self.learning_rate * db
