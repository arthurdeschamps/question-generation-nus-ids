import functools

from training.bert_trainer import Trainer
import tensorflow as tf


class GPTTrainer(Trainer):

    def train_step(self,
                   total_loss,
                   nb_losses,
                   paragraph_tokens,
                   ref_question,
                   step,
                   global_step: tf.Variable):
        preds = [[] for _ in range(ref_question.shape[0])]
        context = paragraph_tokens
        past = None
        types = tf.constant(0, dtype=tf.int32, shape=paragraph_tokens.shape)
        batch_dim = paragraph_tokens.shape[0]

        for i in range(ref_question.shape[1]):
            ref_tokens = ref_question[:, i]
            if ref_tokens[0] == self.embedder.padding_token:
                break
            predictions, past, token_loss = self.token_pred_and_loss(context, past, ref_tokens, types)
            context = tf.expand_dims(ref_question[:, i], axis=1)
            types = tf.constant(1, dtype=tf.int32, shape=(batch_dim, 1))
            pred_index = 0
            for ref_index, ref in enumerate(ref_tokens):
                if ref != self.embedder.padding_token:
                    preds[ref_index].append(predictions[pred_index])
                    pred_index += 1
            total_loss = tf.add(total_loss, token_loss)

        if self.print_predictions:
            for i, pred in enumerate(preds):
                ref = self.embedder.tokenizer.decode(ref_question[i])
                pred = self.embedder.tokenizer.decode(tf.stack(pred))
                paragraph = self.embedder.tokenizer.decode(paragraph_tokens[i])
                tf.print(paragraph, "\n", ref, "\n", pred, "\n")
        global_step.assign(global_step + 1)

        with self.train_summary_writer.as_default():
            total_loss = total_loss / len(preds[0])
            tf.print("Sentence loss: ", total_loss)
            tf.summary.scalar('loss', total_loss, step=global_step)

    @tf.function
    def test_step(self, paragraph_tokens, target_question_tokens, global_step, log_metrics):
        context = paragraph_tokens
        past = None
        token_type_ids = tf.constant(0, shape=paragraph_tokens.shape)
        predicted_tokens = []
        self.test_loss.reset_states()
        for i in tf.range(target_question_tokens.shape[0]):
            all_logits, past = self.model.model(context, past=past, token_type_ids=token_type_ids, training=False)
            logits = all_logits[0, -1]
            context = tf.reshape(tf.argmax(tf.math.softmax(logits)), shape=(1, 1))
            predicted_tokens.append(context)
            token_type_ids = tf.constant(1, shape=(1, 1), dtype=tf.int32)
            loss = self.train_loss_object(target_question_tokens[i], logits)
            self.test_loss(loss)

        def compute_accuracy(target_question, generated_question):
            target_question = self.embedder.vocab_lookup(target_question.numpy())
            generated_question = self.embedder.vocab_lookup(generated_question.numpy()).replace('?', '')
            acc = self.test_accuracy([generated_question], [target_question])
            if log_metrics:
                with self.test_summary_writer.as_default():
                    tf.summary.scalar('dev_accuracy', acc, step=global_step)
            return acc

        tf.py_function(
            compute_accuracy,
            inp=[target_question_tokens, predicted_tokens],
            Tout=tf.float32
        )
        if log_metrics:
            with self.test_summary_writer.as_default():
                tf.summary.scalar('dev_loss', self.test_loss.result(), step=global_step)

    @tf.function(experimental_relax_shapes=True)
    def token_pred_and_loss(self, context, past, target, token_type_ids):
        with tf.GradientTape() as tape:
            mask = tf.cast(tf.not_equal(context, self.embedder.padding_token),
                           dtype=tf.int32, name="attention_mask")
            all_logits, past = self.model.model(
                context,
                past=past,
                attention_mask=mask,
                token_type_ids=token_type_ids,
                training=True
            )
            logits = all_logits[:, -1]
            padding_free_indices = tf.where(tf.not_equal(target, tf.fill(
                target.shape, value=tf.constant(self.embedder.tokenizer.pad_token_id, tf.int32) )))
            padding_free_logits = tf.gather(logits, padding_free_indices)
            padding_free_predictions = tf.math.argmax(tf.math.softmax(padding_free_logits, axis=-1), axis=-1, output_type=tf.int32)
            padding_free_targets = tf.gather(target, padding_free_indices)
            # Accuracy
            self.train_accuracy(padding_free_targets, padding_free_predictions)
            has_non_padding_outputs = tf.greater(tf.size(padding_free_targets), tf.constant(0))

            def apply_gradient():
                loss = tf.reduce_mean(self.train_loss_object(padding_free_targets, padding_free_logits))
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                return loss

            loss = tf.cond(has_non_padding_outputs, apply_gradient, lambda: 0.0)

        return padding_free_predictions, past, loss
