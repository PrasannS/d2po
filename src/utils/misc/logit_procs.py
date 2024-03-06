        
from transformers import LogitsProcessor

class EinsteinLogitsProc(LogitsProcessor):
    
    def __init__(self, tokenizer, max_tokens_without_comma=4, max_newlines=1):
        self.tokenizer = tokenizer
        self.max_tokens_without_comma = max_tokens_without_comma
        self.max_newlines = max_newlines
        # Initialize counters for each sequence in the batch
        self.tokens_since_last_comma = [0]
        self.last_token_was_comma = [False]
        self.newline_counts = [0]
        self.initial = True

    def __call__(self, input_ids, scores):
        # Identify the token ID for comma
        comma_token_id = self.tokenizer.convert_tokens_to_ids(',')
        
        # Adjust counters for batch size changes
        batch_size = input_ids.shape[0]
        self._adjust_counters_for_batch_size(batch_size)

        for i in range(batch_size):
            # Convert last token to string
            last_token_str = self.tokenizer.decode(input_ids[i, -1])

            # Update counters based on the last token
            self._update_counters(i, last_token_str, comma_token_id)
            
            # Boost or suppress logits based on conditions
            scores = self._modify_logits(i, scores, last_token_str, comma_token_id)

        return scores

    def _adjust_counters_for_batch_size(self, batch_size):
        diff = batch_size - len(self.tokens_since_last_comma)
        if diff > 0:
            self.tokens_since_last_comma.extend([0] * diff)
            self.last_token_was_comma.extend([False] * diff)
            self.newline_counts.extend([0] * diff)

    def _update_counters(self, i, last_token_str, comma_token_id):
        if ',' in last_token_str:
            self.tokens_since_last_comma[i] = 0
            self.last_token_was_comma[i] = True
        else:
            self.tokens_since_last_comma[i] += 1
            self.last_token_was_comma[i] = False

        if '\n' in last_token_str:
            self.newline_counts[i] += 1

    def _modify_logits(self, i, scores, last_token_str, comma_token_id):
        # Boost comma logits if needed
        if self.tokens_since_last_comma[i] >= self.max_tokens_without_comma:
            scores[i, comma_token_id] += 1000  # Arbitrary large value

        # Suppress comma logits if the last token was a comma
        if self.last_token_was_comma[i]:
            scores[i, comma_token_id] = -float('inf')

        # Suppress newline logits if the max number of newlines has been reached
        newline_token_id = self.tokenizer.convert_tokens_to_ids('\n')
        if self.newline_counts[i] >= self.max_newlines:
            scores[i, newline_token_id] = -float('inf')

        return scores