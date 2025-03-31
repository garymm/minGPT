"""
Trains a GPT on the WikiText-2 dataset from Hugging Face.
"""

import os
import sys
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------


def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = "./out/wikitext"

    # data
    C.data = WikiTextDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = "gpt2-xl"  # Start with a smaller model for faster training

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
    of block_size tokens for training.
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128  # Context size for the model
        C.dataset_name = "wikitext"
        C.dataset_config = "wikitext-2-raw-v1"
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

        # Build vocabulary from dataset
        self.chars = sorted(list(set("".join(self.dataset["text"]))))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        # Process dataset into token sequences
        self.data = self._prepare_data()

    def _prepare_data(self):
        """Process the dataset into token sequences."""
        # Concatenate all texts in the dataset
        all_text = " ".join(self.dataset["text"])

        # Encode the text as integers
        data = torch.tensor([self.stoi[c] for c in all_text], dtype=torch.long)
        return data

    def get_vocab_size(self):
        return len(self.chars)

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
    def eval_split(trainer, split):
        dataset = {"train": train_dataset, "test": test_dataset}[split]
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)

        model.eval()
        losses = []
        for x, y in loader:
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
            context = torch.tensor(
                [[train_dataset.stoi[c] for c in prompt]], dtype=torch.long
            ).to(trainer.device)
        else:
            # Start with a random seed from the dataset
            idx = torch.randint(0, len(train_dataset), (1,))
            context, _ = train_dataset[idx]
            context = context.unsqueeze(0).to(trainer.device)

        # Generate tokens
        generated = model.generate(
            context, max_new_tokens=max_tokens, do_sample=True, temperature=0.8
        )

        # Convert back to text
        generated_chars = [train_dataset.itos[int(i)] for i in generated[0]]
        return "".join(generated_chars)

    # iteration callback
    best_loss = float("inf")

    def batch_end_callback(trainer):
        global best_loss

        if trainer.iter_num % 100 == 0:
            print(
                f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}"
            )

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            model.eval()
            with torch.no_grad():
                test_loss = eval_split(trainer, "test")

                # Generate some sample text
                print("\nSample text generation:")
                sample_text = generate_text(model, max_tokens=50)
                print(sample_text)
                print("-" * 50)

            # save the model if this is the best score we've seen so far
            if test_loss < best_loss:
                best_loss = test_loss
                print(f"saving model with new best loss of {test_loss:.4f}")
                ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                torch.save(model.state_dict(), ckpt_path)

            # revert model to training mode
            model.train()

    trainer.set_callback("on_batch_end", batch_end_callback)

    # run the optimization
    trainer.run()
