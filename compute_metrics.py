class ModelHandler:
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids)[0]

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Compute unnormalized loss for each token in the batch
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        unnormalized_lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        unnormalized_lm_loss = unnormalized_lm_loss.view(shift_labels.size())

        # Compute average loss per token for each sequence (normalized_lm_loss)
        sequence_lengths = shift_labels.ne(0).sum(dim=1).float()
        normalized_lm_loss = unnormalized_lm_loss.sum(dim=1) / sequence_lengths  # Also known as lm_loss_per_sequence

        # Compute the total loss for the batch
        total_batch_loss = unnormalized_lm_loss.sum()

        # Compute the mean loss per sequence in the batch
        mean_lm_loss_per_sequence = normalized_lm_loss.mean()

        # Compute perplexity based on the total loss for the batch
        perplexity = torch.exp(total_batch_loss / sequence_lengths.sum())

        return {
            "unnormalized_lm_loss": unnormalized_lm_loss,
            "normalized_lm_loss": normalized_lm_loss,
            "total_batch_loss": total_batch_loss,
            "mean_lm_loss_per_sequence": mean_lm_loss_per_sequence,
            "perplexity": perplexity
        }

# Example usage
# model_handler = ModelHandler()
# loss_metrics = model_handler.compute_loss(model, inputs)
