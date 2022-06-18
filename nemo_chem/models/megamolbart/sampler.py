import torch
from torch import nn


class TopkDecoder(torch.nn.Module):
    """
    Top-k decoding functions for sampling
    """

    def __init__(self, tokenizer, max_seq_len):
        super(TopkDecoder, self).__init__()

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def forward(self, decode_fn, batch_size, tokens_enc, enc_masks, top_k=1, temp=1, device='cpu'):
        predicted_seqs = (
            torch.LongTensor([self.tokenizer.bos_id] * tokens_enc.size(0)).unsqueeze(1).to(device)
        )
        predicted_seq_probs = torch.zeros((batch_size * top_k, 1),
                                          device=device,
                                          dtype=torch.float)

        first = True
        hidden_states = None

        #TODO: Break once eos is reached for all samples
        for _ in range(self.max_seq_len):
            token_logits, hidden_states, enc_output_masks = \
                decode_fn(predicted_tokens_dec=predicted_seqs,
                          tokens_enc=tokens_enc,
                          enc_masks=enc_masks)

            # Resize tensors for holding temp values
            predicted_seqs = predicted_seqs.repeat_interleave(top_k, 0)
            predicted_seq_probs = predicted_seq_probs.repeat_interleave(top_k, 0)

            if first:
                enc_masks = enc_masks.repeat_interleave(top_k, 0)
                tokens_enc = tokens_enc.repeat_interleave(top_k, 0)
                first = False

            # Get top-k for each item in batch
            log_probs, token_ids = torch.topk(
                nn.functional.log_softmax(token_logits, dim=-1),
                top_k)
            log_probs = (log_probs / temp).softmax(log_probs.dim() - 1)

            # For each sequence add the predicted token to create seq for next step
            next_seq_probs = None
            next_seqs = None

            # Iterate for each input row in the original batch.
            for i in range(batch_size):
                item_size = int(predicted_seqs.shape[0] / batch_size)
                item_start_pos = i * item_size
                item_end_pos = item_start_pos + item_size

                # Extract part of seq and probs from the final result for the batch
                part_seqs = predicted_seqs[item_start_pos:item_end_pos]
                part_seq_probs = predicted_seq_probs[item_start_pos:item_end_pos]

                # Separate decoded values by the original input values.
                item_size = int(log_probs.shape[0] / batch_size)
                item_start_pos = i * item_size
                item_end_pos = item_start_pos + item_size
                # Extract part of seq and probs for the current step
                part_token_ids = token_ids[item_start_pos:item_end_pos, 0]

                part_log_probs = log_probs[item_start_pos:item_end_pos, 0]

                temp_seqs = torch.cat((part_seqs, part_token_ids.flatten().T.unsqueeze(1)), dim=1)
                temp_probs, idx = torch.topk(part_log_probs.flatten(), top_k)
                temp_seqs = torch.index_select(temp_seqs, 0, idx)

                if next_seqs is None:
                    next_seqs = temp_seqs
                    next_seq_probs = temp_probs
                else:
                    next_seqs = torch.cat((next_seqs, temp_seqs), dim=0)
                    next_seq_probs = torch.cat((next_seq_probs, temp_probs), dim=0)

            predicted_seqs = next_seqs
            predicted_seq_probs = next_seq_probs

        predicted_tokens_ids = predicted_seqs.cpu().detach().numpy().tolist()
        for item, predicted_tokens_ in enumerate(predicted_tokens_ids):
            if self.tokenizer.eos_id in predicted_tokens_:
                idx = predicted_tokens_.index(self.tokenizer.eos_id)
                predicted_tokens_ids[item] = predicted_tokens_[:idx]
            else:
                predicted_tokens_ids[item] = \
                    [id for id in predicted_tokens_ if id != self.tokenizer.pad_id]

        # predicted_tokens_text = self.tokenizer.ids_to_tokens(predicted_tokens_ids)
        # sampled_smiles = self.tokenizer.tokens_to_text(predicted_tokens_text)
        sampled_smiles = self.tokenizer.ids_to_text(predicted_tokens_ids)

        return sampled_smiles, hidden_states, enc_output_masks