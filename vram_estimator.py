from optimizers import get_memory_factor


def estimate_vram_usage(
    model_size,
    precision,
    sequence_length,
    batch_size,
    hidden_size,
    num_layers,
    optimizer,
):
    """
    Estimate the VRAM required for fine-tuning a model.

    Parameters:
        model_size (int): Number of parameters in the model (e.g., 7 billion = 7e9).
        precision (int): Bytes per parameter (e.g., 2 for FP16, 4 for FP32).
        sequence_length (int): Length of input sequences.
        batch_size (int): Number of samples per batch.
        hidden_size (int): Hidden size of the model.
        num_layers (int): Number of layers in the model.
        optimizer (str): Optimizer type ('adam', 'adamw', 'sgd', 'rmsprop', 'adafactor').

    Returns:
        float: Estimated VRAM usage in GB.
    """

    # Model Weights
    model_weight_memory = model_size * precision

    # Activations (approximate)
    activation_memory = (
        batch_size * sequence_length * hidden_size * num_layers * precision
    )

    # Gradients (similar size to activations)
    gradient_memory = activation_memory

    # Optimizer States
    optimizer_factor = get_memory_factor(optimizer)
    optimizer_memory = optimizer_factor * model_weight_memory

    # Total Memory
    total_memory = (
        model_weight_memory + activation_memory + gradient_memory + optimizer_memory
    )

    # Convert to GB
    total_memory_gb = total_memory / (1024**3)
    return total_memory_gb
