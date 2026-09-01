import logging

def _modify_embedding_weights(
    model,
    embeddings,
    previous_embedding_weights,
    old_tokenizer,
    new_tokenizer,
    previous_embedding_bias=None,
    init_method=None,
    ignore_size_mismatch=False
):
    old_vocab = old_tokenizer.get_vocab()
    new_vocab = new_tokenizer.get_vocab()
    new_vocab_reverse = {idx: token for token, idx in new_tokenizer.get_vocab().items()}

    has_bias = getattr(embeddings, "bias", None) is not None

    if has_bias and previous_embedding_bias is None:
        raise ValueError("Bias not provided.")
    if not has_bias and previous_embedding_bias is not None:
        raise ValueError("Bias provided for embedding without bias")
    if not ignore_size_mismatch and len(new_vocab) != embeddings.weight.size(0):
        raise ValueError(
            f"Embedding size ({embeddings.weight.size(0)}) does not match with vocabulary size ({len(new_vocab)}) in the new vocabulary."
        )
    if not ignore_size_mismatch and len(old_vocab) != previous_embedding_weights.size(0):
        raise ValueError(f"Embedding size ({previous_embedding_weights.size(0)}) does not match with vocabulary size ({len(old_vocab)}) in the old vocabulary.")

    operations = []
    uninitialized_positions = set(range(embeddings.weight.size(0)))

    # Randomly initializing all weights according to the model configuration
    logging.debug(f"Initializing tokens")
    model._init_weights(embeddings)

    # Copying weights from overlapping weights
    for token, idx in new_vocab.items():
        if token not in old_vocab:
            continue
        uninitialized_positions.remove(idx)

        old_idx = old_vocab[token]
        logging.debug(f"Copying weight {old_idx} -> {idx} ({token})")
        operations.append({"type": "copy", "prev_idx": old_idx, "idx": idx, "token": token})

        embeddings.weight.data[idx] = previous_embedding_weights[old_idx]
        if has_bias:
            logging.debug(f"Copying bias {old_idx} -> {idx} ({token})")
            embeddings.bias.data[idx] = previous_embedding_bias[old_idx]

    # Initializing uninitialized weights
    if init_method is None or init_method == "random":
        # new positionas are already randomly initialized
        operations.extend([{"type": "initialize", "method": "random", "idx": idx, "token": new_vocab_reverse[idx]} for idx in uninitialized_positions])
    elif init_method == "mean":
        mean_embedding = previous_embedding_weights.mean(axis=0)
        mean_bias = previous_embedding_bias.mean(axis=0) if has_bias else None

        for idx in uninitialized_positions:
            token = new_vocab_reverse[idx]
            operations.append({"type": "initialize", "method": init_method, "idx": idx, "token": token})
            logging.debug(f"Initializing weight {idx} ({token}) with {init_method}")

            embeddings.weight.data[idx] = mean_embedding
            if has_bias:
                logging.debug(f"Initializing bias {idx} ({token}) with {init_method}")
                embeddings.bias.data[idx] = mean_bias
    elif init_method == "mean_of_constituents":
        for idx in uninitialized_positions:
            token = new_vocab_reverse[idx]
            constituent_ids = [token.id for token in old_tokenizer._tokenizer.model.tokenize(token)]
            embeddings.weight.data[idx] = previous_embedding_weights[constituent_ids, :].mean(axis=0)
            operations.append({"type": "initialize", "method": init_method, "idx": idx, "token": token, "constituents": constituent_ids})
            logging.debug(f"Initializing weight {idx} ({token}) with {init_method}")
            if has_bias:
                raise NotImplementedError("mean_of_constituents not implemented for models with bias terms for embeddings")
    else:
        raise ValueError("Unknown initialization method")

    return {
        "operations": list(sorted(operations, key=lambda x: x["idx"])),
        "removed_tokens": {token: idx for token, idx in old_vocab.items() if token not in new_vocab},
    }


def modify_embeddings(
    model,
    old_tokenizer,
    new_tokenizer,
    init_method="mean_of_constituents",
    ignore_size_mismatch: bool = False
):
    # Clone previous weights before resizing
    previous_input_embeddings = model.get_input_embeddings().weight.data.clone()
    if model.get_output_embeddings() is not None and not model.config.tie_word_embeddings:
        output_embedding = model.get_output_embeddings()
        previous_output_embeddings = output_embedding.weight.data.clone()
        previous_output_bias = output_embedding.bias.data.clone() if output_embedding.bias is not None else None
    else:
        previous_output_embeddings = None
        previous_output_bias = None

    modification_log = {}

    logging.info("Resizing embeddings")
    model.resize_token_embeddings(len(new_tokenizer))

    logging.info("Modifying input embeddings")
    input_embeddings = model.get_input_embeddings()
    modification_log["input"] = _modify_embedding_weights(
        model,
        input_embeddings,
        previous_input_embeddings,
        old_tokenizer,
        new_tokenizer,
        init_method=init_method,
        ignore_size_mismatch=ignore_size_mismatch,
    )

    if previous_output_embeddings is not None:
        logging.info("Modifying output embeddings")
        output_embeddings = model.get_output_embeddings()
        modification_log["output"] = _modify_embedding_weights(
            model,
            output_embeddings,
            previous_output_embeddings,
            old_tokenizer,
            new_tokenizer,
            init_method=init_method,
            previous_embedding_bias=previous_output_bias,
            ignore_size_mismatch=ignore_size_mismatch,
        )

    model.tie_weights()

    if "output" in modification_log:
        assert modification_log["input"]["operations"] == modification_log["output"]["operations"]
        assert modification_log["input"]["removed_tokens"] == modification_log["output"]["removed_tokens"]

    n_copied_tokens = len([operation for operation in modification_log["input"]["operations"] if operation["type"] == "copy"])
    n_removed_tokens = previous_input_embeddings.size(0) - n_copied_tokens
    n_new_tokens = len(new_tokenizer) - n_copied_tokens
    logging.info(f"Kept {n_copied_tokens} tokens, added {n_new_tokens} tokens, Removed {n_removed_tokens} tokens")

    return modification_log["input"]