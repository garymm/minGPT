"""
Trains a GPT on the WikiText-2 dataset from Hugging Face.
"""

import os
import sys
import time
from datetime import datetime

import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import CfgNode as CN
from mingpt.utils import set_seed, setup_logging

# -----------------------------------------------------------------------------


def get_config():
    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = "./out/wikitext"
    C.system.profile = False  # Profile the training process
    C.system.eval_every = 500  # Evaluate the model every N iterations

    # data
    C.data = WikiTextDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = "gpt2-xl"

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 3e-4
    C.trainer.max_iters = 5000
    C.trainer.batch_size = 32

    return C


# -----------------------------------------------------------------------------


class WikiTextDataset(Dataset):
    """
    Dataset for WikiText-2 from Hugging Face. Processes the text into chunks
    of block_size tokens for training using a standard HuggingFace tokenizer.
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128  # Context size for the model
        C.dataset_name = "wikitext"
        C.dataset_config = "wikitext-2-v1"  # TODO: switch to wikitext-103-v1
        C.tokenizer_name = "gpt2"
        return C

    def __init__(self, config, split):
        self.config = config
        self.split = split  # train/test/validation

        # Map 'test' to 'validation' for Hugging Face's wikitext dataset which uses that naming
        hf_split = "validation" if split == "test" else split

        # Load dataset from Hugging Face
        self.dataset = load_dataset(
            self.config.dataset_name, self.config.dataset_config, split=hf_split
        )

        # Load the tokenizer from HuggingFace
        self.tokenizer = Tokenizer.from_pretrained(self.config.tokenizer_name)

        # Process dataset into token sequences
        self.data = self._prepare_data()

    def _prepare_data(self):
        """Process the dataset into token sequences using the tokenizer."""
        # Concatenate all texts in the dataset
        all_text = " ".join(self.dataset["text"])

        # Tokenize the entire text
        encodings = self.tokenizer.encode(all_text)

        # Extract token IDs as a flat tensor
        return torch.tensor(encodings.ids, dtype=torch.long).squeeze()

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size

    def __getitem__(self, idx):
        # Grab a chunk of data at position idx
        chunk = self.data[idx : idx + self.config.block_size + 1]
        x = chunk[:-1]  # Input sequence
        y = chunk[1:]  # Target sequence (shifted by 1)
        return x, y


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct train and test datasets
    train_dataset = WikiTextDataset(config.data, split="train")
    test_dataset = WikiTextDataset(config.data, split="test")

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # helper function for evaluation
    def eval_split(trainer, split, max_batches=None):
        dataset = {"train": train_dataset, "test": test_dataset}[split]
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)

        model.eval()
        losses = []
        for x, y in loader:
            if max_batches and len(losses) >= max_batches:
                break
            x, y = x.to(trainer.device), y.to(trainer.device)
            with torch.no_grad():
                logits, loss = model(x, y)
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        print(f"{split} loss: {avg_loss:.4f}")
        return avg_loss

    # helper function for text generation
    def generate_text(model, prompt="", max_tokens=100):
        model.eval()

        # Convert prompt to token ids if provided
        if prompt:
            # Use the tokenizer to encode the prompt
            context = train_dataset.tokenizer(prompt, return_tensors="pt").input_ids.to(
                trainer.device
            )
        else:
            # Start with a random seed from the dataset
            idx = torch.randint(0, len(train_dataset), (1,))
            context, _ = train_dataset[idx]
            context = context.unsqueeze(0).to(trainer.device)

        # Generate tokens
        generated = model.generate(
            context, max_new_tokens=max_tokens, do_sample=True, temperature=0.8
        )

        # Convert back to text using the tokenizer
        generated_text = train_dataset.tokenizer.decode(
            list(generated[0]), skip_special_tokens=True
        )
        return generated_text

    # iteration callback
    best_loss = float("inf")

    def batch_end_callback(trainer):
        global best_loss

        if trainer.iter_num % 100 == 0:
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"
            )

        if (
            config.system.eval_every
            and trainer.iter_num % config.system.eval_every == 0
        ):
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                test_loss = eval_split(trainer, "test", max_batches=100)

                # Generate some sample text
                print("\nSample text generation:")
                sample_text = generate_text(model, max_tokens=50)
                print(sample_text)
                print("-" * 50)

            # revert model to training mode
            model.train()

    eval_split(trainer, "test", max_batches=1)
    # Calculate total tokens that will be processed
    tokens_per_batch = config.trainer.batch_size * config.data.block_size
    start_time = time.time()

    # warm up
    if config.system.profile:
        from torch.profiler import ProfilerActivity, profile, record_function

        # Get timestamp for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Configure the profiler
        torch.cuda.memory._record_memory_history()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            # Breaks because of https://github.com/pytorch/pytorch/issues/146900
            with_stack=False,
        ) as prof:
            with record_function("trainer.run"):
                trainer.run()

        end_time = time.time()
        total_time = end_time - start_time
        total_tokens = tokens_per_batch * trainer.iter_num
        tokens_per_second = total_tokens / total_time

        # Print and save profiling results
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
        print(f"Tokens per second: {tokens_per_second:,.2f}")

        # Save results
        prof_path = os.path.join(config.system.work_dir, "profile_results")
        os.makedirs(prof_path, exist_ok=True)

        # Save trace with timestamp
        trace_file = os.path.join(prof_path, f"trace_{timestamp}.json")
        prof.export_chrome_trace(trace_file)

        # Save metrics with timestamp
        metrics_file = os.path.join(prof_path, f"metrics_{timestamp}.txt")
        with open(metrics_file, "w") as f:
            f.write("Profile Results\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total tokens processed: {total_tokens:,}\n")
            f.write(f"Total time: {total_time:.2f} seconds\n")
            f.write(f"Tokens per second: {tokens_per_second:,.2f}\n")
            f.write("\nConfiguration:\n")
            f.write(f"Batch size: {config.trainer.batch_size}\n")
            f.write(f"Block size: {config.data.block_size}\n")
            f.write(f"Total iterations: {trainer.iter_num}\n")

        torch.cuda.memory._dump_snapshot(
            os.path.join(prof_path, f"memory_timeline_{timestamp}.pkl")
        )

        print(f"Profiling results saved to: {prof_path}")

    else:
        trainer.set_callback("on_batch_end", batch_end_callback)
        trainer.run()
        end_time = time.time()
        total_time = end_time - start_time
        total_tokens = tokens_per_batch * trainer.iter_num
        tokens_per_second = total_tokens / total_time
        print(f"Tokens per second: {tokens_per_second:,.2f}")
