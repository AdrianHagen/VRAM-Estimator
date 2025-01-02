import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from vram_estimator import estimate_vram_usage
from optimizers import Optimizers, get_optimizer


device = "cpu"


def get_actual_vram_usage(
    model_name, optimizer_enum, batch_size=32, num_passes=1, data_path="data/text.txt"
):
    """
    Get the actual VRAM usage during fine-tuning.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    optimizer = get_optimizer(optimizer_enum, model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sequence_length = model.config.max_position_embeddings
    text = open(data_path, "r").read()

    # Tokenizing and preparing input text for the model
    text = tokenizer(
        [text] * batch_size,
        return_tensors="pt",
        max_length=sequence_length,
        truncation=True,
    ).to(device)

    peak_memory_list = []

    for i in range(num_passes):
        labels = text["input_ids"]
        outputs = model(**text, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3  # GB

        peak_memory_list.append(peak_memory)

    mean_peak_memory = np.mean(peak_memory_list) / num_passes

    return mean_peak_memory


def evaluate_estimator(estimator, model_name, optimizer, precision=4, batch_size=32):
    """
    Evaluate the estimator on a single model.
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    sequence_length = model.config.max_position_embeddings
    hidden_size = model.config.n_embd  # Hidden size
    num_layers = model.config.n_layer  # Number of layers
    estimated_vram = estimator(
        model_size,
        precision,
        sequence_length,
        batch_size,
        hidden_size,
        num_layers,
        optimizer,
    )
    actual_vram = get_actual_vram_usage(model_name, optimizer)
    print(f"Estimated VRAM: {estimated_vram}")
    print(f"Actual VRAM: {actual_vram}")
    print(f"Diff: {abs(estimated_vram - actual_vram)}")
    return abs(estimated_vram - actual_vram)
