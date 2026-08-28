"""Adapt model embeddings to a trained or pruned tokenizer."""

from typing import Any, Dict, Optional

import torch


def adapt_token_embeddings(
    model: Any,
    old_tokenizer: Any,
    new_tokenizer: Any,
    method: str,
    pad_to_multiple_of: Optional[int] = None,
) -> Dict[str, int]:
    """Resize and initialize embeddings using actual vocabulary changes."""
    if method not in {"random", "mean", "mean_of_constituents"}:
        raise ValueError(
            "embedding_init_method must be 'random', 'mean', or "
            "'mean_of_constituents'"
        )
    if pad_to_multiple_of is not None and pad_to_multiple_of <= 0:
        raise ValueError("embedding_pad_to_multiple_of must be positive")

    old_vocab = old_tokenizer.get_vocab()
    new_vocab = new_tokenizer.get_vocab()
    retained_tokens = old_vocab.keys() & new_vocab.keys()
    added_tokens = new_vocab.keys() - old_vocab.keys()
    removed_tokens = old_vocab.keys() - new_vocab.keys()
    moved_tokens = {
        token
        for token in retained_tokens
        if old_vocab[token] != new_vocab[token]
    }

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    current_rows = input_embeddings.weight.shape[0]

    if not added_tokens and not removed_tokens and not moved_tokens:
        return {
            "previous_rows": current_rows,
            "final_rows": current_rows,
            "added_tokens": 0,
            "removed_tokens": 0,
            "moved_tokens": 0,
        }

    old_input = input_embeddings.weight.detach().clone()
    tied = (
        output_embeddings is None
        or output_embeddings is input_embeddings
        or output_embeddings.weight.data_ptr()
        == input_embeddings.weight.data_ptr()
    )
    old_output = (
        None if tied else output_embeddings.weight.detach().clone()
    )
    old_output_bias = (
        None
        if tied or getattr(output_embeddings, "bias", None) is None
        else output_embeddings.bias.detach().clone()
    )

    final_vocab_size = (
        len(old_vocab) + len(added_tokens) - len(removed_tokens)
    )
    if final_vocab_size != len(new_vocab):
        raise ValueError("Tokenizer vocabulary accounting is inconsistent")

    target_rows = (
        final_vocab_size
        if removed_tokens
        else max(current_rows, final_vocab_size)
    )
    if target_rows != current_rows:
        model.resize_token_embeddings(
            target_rows,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings.weight.shape[0] < final_vocab_size:
        raise RuntimeError("Embedding resize did not provide tokenizer capacity")

    old_ids = sorted(
        token_id
        for token_id in set(old_vocab.values())
        if 0 <= token_id < old_input.shape[0]
    )
    old_id_set = set(old_ids)
    input_mean = old_input[old_ids].float().mean(dim=0).to(old_input.dtype)
    output_mean = (
        None
        if old_output is None
        else old_output[old_ids].float().mean(dim=0).to(old_output.dtype)
    )
    initializer_range = float(
        getattr(model.config, "initializer_range", 0.02)
    )

    def constituent_ids(token: str):
        tokenizer_model = getattr(
            getattr(old_tokenizer, "_tokenizer", None), "model", None
        )
        if tokenizer_model is None:
            return []
        return [
            piece.id
            for piece in tokenizer_model.tokenize(token)
            if piece.id in old_id_set
        ]

    def initialized_value(source, mean_value, token, target_row):
        if method == "random":
            value = torch.empty_like(target_row, dtype=torch.float32)
            value.normal_(mean=0.0, std=initializer_range)
            return value.to(target_row.dtype)
        if method == "mean":
            return mean_value

        ids = constituent_ids(token)
        if not ids:
            return mean_value
        return source[ids].float().mean(dim=0).to(source.dtype)

    with torch.no_grad():
        for token in retained_tokens:
            old_id = old_vocab[token]
            new_id = new_vocab[token]
            input_embeddings.weight[new_id].copy_(old_input[old_id])

            if old_output is not None:
                output_embeddings.weight[new_id].copy_(old_output[old_id])
                if old_output_bias is not None:
                    output_embeddings.bias[new_id].copy_(
                        old_output_bias[old_id]
                    )

        for token in added_tokens:
            new_id = new_vocab[token]
            input_embeddings.weight[new_id].copy_(
                initialized_value(
                    old_input,
                    input_mean,
                    token,
                    input_embeddings.weight[new_id],
                )
            )

            if old_output is not None:
                output_embeddings.weight[new_id].copy_(
                    initialized_value(
                        old_output,
                        output_mean,
                        token,
                        output_embeddings.weight[new_id],
                    )
                )
                if getattr(output_embeddings, "bias", None) is not None:
                    output_embeddings.bias[new_id].zero_()

    if tied:
        model.tie_weights()

    return {
        "previous_rows": current_rows,
        "final_rows": model.get_input_embeddings().weight.shape[0],
        "added_tokens": len(added_tokens),
        "removed_tokens": len(removed_tokens),
        "moved_tokens": len(moved_tokens),
    }
