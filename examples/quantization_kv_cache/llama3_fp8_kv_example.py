from typing import Callable, Any, Type, List, Set
from functools import partial
import matplotlib.pyplot as plt

import gc
import torch
import weakref


class TrackTensorAllocations:
    total_tensor_memory: int
    memory_timeline: List[int]
    
    _tracked_tensors: Set[int]
    _original_init_fn: Callable[[Any], None]

    def __init__(self):
        self.total_tensor_memory = 0
        self.memory_timeline = []

        self._tracked_tensors = set()
        self._original_init_fn = torch.Tensor.__init__

    def __enter__(self):
        def wrapped_init(instance, *args, **kwargs):
            if isinstance(instance, torch.Tensor):
                self._original_init_fn(instance)
                self.track_tensor(instance)
            else:
                # parameters, ect.
                type(instance).__init__(instance, *args, **kwargs)
        
        torch.Tensor.__init__ = wrapped_init

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.Tensor.__init__ = self._original_init_fn
        self._active = False
        gc.collect()

    def track_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor_hash = hash(tensor)
        tensor_memory = tensor.numel() * tensor.element_size()

        # warn when init is called twice
        if tensor_hash in self._tracked_tensors:
            print("double init")
            return

        # add memory
        self.total_tensor_memory += tensor_memory
        self._add_to_timeline()
        self._tracked_tensors.add(tensor_hash)

        # register hook to subtract memory
        weakref.finalize(tensor, partial(self._on_tensor_deallocated, tensor_memory, tensor_hash))

    def _on_tensor_deallocated(self, tensor_memory, tensor_hash):
        self.total_tensor_memory -= tensor_memory
        self._add_to_timeline()
        self._tracked_tensors.remove(tensor_hash)
    
    @property
    def total_tensor_memory_mib(self):
        return self.total_tensor_memory / (1024 * 1024)
    
    def _add_to_timeline(self):
        self.memory_timeline.append(self.total_tensor_memory)

    def plot_values_over_time(self, dpi=300):
        values = self.memory_timeline
        """
        Plots a list of float values over time using matplotlib.

        Parameters:
            values (list of float): The values to plot.
        """
        if not values:
            print("The list of values is empty.")
            return

        plt.figure(figsize=(10, 4))
        plt.plot(range(len(values)), values, marker='o', linestyle='-')
        plt.title("Values Over Time")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("file.png", dpi=dpi)


with TrackTensorAllocations() as prof:

    from datasets import load_dataset
    from loguru import logger
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor import oneshot

    # Select model and load it.
    MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Select calibration dataset.
    DATASET_ID = "HuggingFaceH4/ultrachat_200k"
    DATASET_SPLIT = "train_sft"

    # Select number of samples. 512 samples is a good place to start.
    # Increasing the number of samples can improve accuracy.
    NUM_CALIBRATION_SAMPLES = 512
    MAX_SEQUENCE_LENGTH = 2048

    # Load dataset and preprocess.
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))


    def process_and_tokenize(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return tokenizer(
            text,
            padding=False,
            max_length=MAX_SEQUENCE_LENGTH,
            truncation=True,
            add_special_tokens=False,
        )


    ds = ds.map(process_and_tokenize, remove_columns=ds.column_names)

    # Configure the quantization algorithm and scheme.
    # In this case, we:
    #   * quantize the weights to fp8 with per-tensor scales
    #   * quantize the activations to fp8 with per-tensor scales
    #   * quantize the kv cache to fp8 with per-tensor scales
    recipe = """
    quant_stage:
        quant_modifiers:
            QuantizationModifier:
                ignore: ["lm_head"]
                config_groups:
                    group_0:
                        weights:
                            num_bits: 8
                            type: float
                            strategy: tensor
                            dynamic: false
                            symmetric: true
                        targets: ["Linear"]
    """

    # Apply algorithms.
    oneshot(
        model=model,
        #dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    logger.info(
        "Running sample generation. ",
        "Note: Inference with the quantized kv_cache is not supported. ",
        "Please use vLLM for inference with the quantized kv_cache.",
    )
    # Confirm generations of the quantized model look sane.
    # print("\n\n")
    # print("========== SAMPLE GENERATION ==============")
    # input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
    # output = model.generate(input_ids, max_new_tokens=100)
    # print(tokenizer.decode(output[0]))
    # print("==========================================\n\n")

    # Save to disk compressed.
    SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-KV"
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)

    breakpoint()