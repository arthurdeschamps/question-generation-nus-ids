import functools

from training.trainer import Trainer
import tensorflow as tf


class GPTTrainer(Trainer):

    def train_step(self,
                   paragraph_tokens,
                   ref_question):

        losses = []
        preds = [[] for _ in range(ref_question.shape[0])]
        context = paragraph_tokens
        past = None
        types = tf.constant(0, dtype=tf.int32, shape=paragraph_tokens.shape)
        batch_dim = paragraph_tokens.shape[0]

        for i in tf.range(ref_question.shape[1]):
            ref_tokens = ref_question[:, i]
            if tf.reduce_any(tf.not_equal(
                    ref_tokens,
                    self.embedder.padding_token
            )):
                predictions, past, token_loss = self.token_pred_and_loss(context, past, ref_tokens, types)
                context = tf.expand_dims(ref_question[:, i], axis=1)
                types = tf.constant(1, dtype=tf.int32, shape=(batch_dim, 1))
                for ind, ref in enumerate(ref_tokens):
                    if ref != self.embedder.padding_token:
                        preds[ind].append(predictions[ind])
                losses.append(token_loss)
        if self.print_predictions:
            for i, pred in enumerate(preds):
                ref = self.embedder.tokenizer.decode(ref_question[i])
                pred = self.embedder.tokenizer.decode(tf.stack(pred))
                paragraph = self.embedder.tokenizer.decode(paragraph_tokens[i])
                tf.print(paragraph, "\n", ref, "\n", pred, "\n")
        total_loss = tf.reduce_mean(losses)
        return total_loss

    @tf.function
    def test_step(self, paragraph_tokens, target_question_tokens, log_metrics):
        context_type_ids = tf.zeros_like(paragraph_tokens, dtype=tf.int32)

        def _produce_distribution(beams):
            token_type_ids = tf.repeat(
                tf.expand_dims(tf.concat((
                    context_type_ids,
                    tf.ones((1, tf.shape(beams)[1] - tf.shape(paragraph_tokens)[1]), dtype=tf.int32)
                ), axis=1), axis=0), repeats=tf.shape(beams)[0], axis=0
            )
            attention_mask = tf.cast(tf.not_equal(beams, self.embedder.padding_token), dtype=tf.int32)
            all_logits, past = self.model.model(
                beams,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                training=False
            )
            logits = all_logits[:, -1]
            return tf.math.softmax(logits, axis=1)

        predicted_tokens = self.model.beam_search(paragraph_tokens, _produce_distribution)

        def without_padding(tokens):
            return tf.reshape(tf.gather(
                tokens, tf.where(tf.not_equal(tokens, self.embedder.padding_token))
            ), shape=(-1,))
        target_question_tokens = without_padding(target_question_tokens)
        predicted_tokens = without_padding(predicted_tokens)
        return target_question_tokens, predicted_tokens

    @tf.function
    def token_pred_and_loss(self, context: tf.Tensor, past, target: tf.Tensor, token_type_ids: tf.Tensor):
        with tf.GradientTape() as tape:
            mask = tf.cast(tf.not_equal(context, self.embedder.padding_token),
                           dtype=tf.int32, name="attention_mask")
            all_logits, past = self.model.model(
                context,
                past=None if past is None else tf.unstack(past, axis=0),
                attention_mask=mask,
                token_type_ids=token_type_ids,
                training=True
            )
            past = tf.stack(past, axis=0)
            logits = all_logits[:, -1]

            predictions = tf.math.argmax(tf.math.softmax(logits, axis=-1, name="train_logits"),
                                         axis=-1, output_type=target.dtype, name="train_predictions")
            mask = tf.cast(tf.not_equal(target, self.embedder.padding_token), dtype=tf.int32, name="loss_mask")
            target = tf.multiply(target, mask, name="masked_target")
            loss = tf.pow(2.0, self.train_loss_object(target, logits))
            # Needs to account for padding tokens
            mask = tf.cast(mask, loss.dtype, name="mask")
            non_padding_count = tf.reduce_sum(mask, name="non_padding_output")

            loss = tf.divide(tf.reduce_sum(tf.multiply(loss, mask, name="masked_loss"),
                                           name="total_masked_loss"), non_padding_count, name="mean_masked_loss")
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            # Only apply gradient if there is at least one non-padding input

        return predictions, past, loss
