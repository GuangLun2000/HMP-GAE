# data_loader.py
# Data loader for text classification (AG News, IMDB, DBpedia, Yahoo Answers) for federated experiments.
# Note: data-agnostic attack setting — no training-time label flipping is performed.

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pandas as pd
import urllib.request
import os
from typing import List, Dict

class NewsDataset(Dataset):
    """Custom Dataset for text classification (AG News, IMDB, DBpedia, Yahoo Answers, etc.).

    Pre-tokenizes the entire text list once at construction so that __getitem__
    is a pure tensor-index op. With Qwen-style fast tokenizers, this is bit-
    identical to per-item tokenization (no RNG, deterministic) but eliminates
    the ~22.5K redundant tokenize calls per FL round that lazy tokenization
    caused. Memory cost is negligible (10K samples × 128 tokens × int64 ≈ 10 MB).
    """

    def __init__(self, texts, labels, tokenizer, max_length=128,
                include_target_mask: bool = False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        # include_target_mask is preserved in the signature for backwards
        # compatibility but is unused in the current codebase (grep confirmed).
        self.include_target_mask = include_target_mask

        # Empty dataset (e.g. DataManager.get_empty_loader() for data-agnostic
        # attackers): build zero-row tensors directly. Calling the
        # fast tokenizer on an empty list raises IndexError on newer
        # transformers (tokens_and_encodings[0][0]), so short-circuit it.
        if len(texts) == 0:
            self._input_ids = torch.empty((0, max_length), dtype=torch.long)
            self._attention_mask = torch.empty((0, max_length), dtype=torch.long)
            self._labels = torch.empty((0,), dtype=torch.long)
            return

        # Batch tokenize once. Same args as the old per-item call, so the
        # produced input_ids / attention_mask are identical.
        encoding = tokenizer(
            [str(t) for t in texts],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
        )
        self._input_ids = encoding['input_ids']         # (N, max_length)
        self._attention_mask = encoding['attention_mask']  # (N, max_length)
        self._labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self):
        return self._input_ids.shape[0]

    def __getitem__(self, idx):
        return {
            'input_ids': self._input_ids[idx],
            'attention_mask': self._attention_mask[idx],
            'labels': self._labels[idx],
        }





class DataManager:
    """Manages text classification data for federated experiments.

    AG News and Yahoo Answers CSVs live under ``data/ag_news/`` and ``data/yahoo_answers/``
    (see ``_load_ag_news`` / ``_load_yahoo_answers``). IMDB and DBpedia load from Hugging Face.
    """

    def __init__(self, num_clients, num_attackers, test_seed,
                 dataset_size_limit=None, batch_size=None, test_batch_size=None,
                 model_name: str = "distilbert-base-uncased", max_length: int = 128,
                 dataset: str = "ag_news"):
        
        """
        Initialize DataManager.
        
        Args:
            num_clients: Number of federated learning clients (required)
            num_attackers: Number of attacker clients (required)
            test_seed: Random seed for test sampling (required)
            dataset_size_limit: Limit dataset size (None = full dataset). For paper reproduction, use None.
                               When set, only limits training set; test set remains full for fair evaluation.
            batch_size: Batch size for training data loaders (required)
            test_batch_size: Batch size for test/validation data loaders (required)
            model_name: Hugging Face model name for tokenizer initialization
            max_length: Max token length (AG News: 128, IMDB: 256-512, DBpedia: 512, Yahoo Answers: 256)
            dataset: 'ag_news' | 'imdb' | 'dbpedia' | 'yahoo_answers'. For ``ag_news`` / ``yahoo_answers``,
                     CSVs are read from ``data/ag_news/`` and ``data/yahoo_answers/`` (see ``data_loader.py``).
        """

        if batch_size is None or test_batch_size is None:
            raise ValueError("batch_size and test_batch_size must be provided via config (see main.py).")

        self.num_clients = num_clients
        self.num_attackers = num_attackers
        self.test_seed = test_seed
        self.dataset_size_limit = dataset_size_limit
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.max_length = max_length
        self.model_name = model_name
        self.dataset = dataset.lower()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Handle padding for decoder-only models (GPT-style)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                print(f"  📝 Set pad_token = eos_token ('{self.tokenizer.eos_token}') for {model_name}")
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                print(f"  📝 Added new pad_token '[PAD]' for {model_name}")

        if self.dataset == "imdb":
            print("Loading IMDB dataset (stanfordnlp/imdb)...")
        elif self.dataset == "dbpedia":
            print("Loading DBpedia dataset (fancyzhx/dbpedia_14)...")
        elif self.dataset == "yahoo_answers":
            print("Loading Yahoo Answers dataset (yassiracharki/Yahoo_Answers_10_categories_for_NLP)...")
        else:
            print("Loading AG News dataset...")
        self._load_data()

    def _load_data(self):
        """Dispatch to dataset-specific loader."""
        if self.dataset == "imdb":
            self._load_imdb()
        elif self.dataset == "dbpedia":
            self._load_dbpedia()
        elif self.dataset == "yahoo_answers":
            self._load_yahoo_answers()
        else:
            self._load_ag_news()

    def _load_imdb(self):
        """Load IMDB dataset from Hugging Face (stanfordnlp/imdb)."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("IMDB requires datasets library. Install: pip install datasets")

        ds = load_dataset("stanfordnlp/imdb")
        train_data = ds["train"]
        test_data = ds["test"]

        self.train_texts = [str(x) for x in train_data["text"]]
        self.train_labels = list(train_data["label"])
        self.test_texts = [str(x) for x in test_data["text"]]
        self.test_labels = list(test_data["label"])

        print(f"  📊 Full IMDB Dataset: Train={len(self.train_texts)}, Test={len(self.test_texts)}")

        if self.dataset_size_limit is not None and self.dataset_size_limit > 0:
            rng = np.random.default_rng(42)
            n_train = min(self.dataset_size_limit, len(self.train_texts))
            n_test = min(int(self.dataset_size_limit * 0.15), len(self.test_texts))
            idx_train = rng.choice(len(self.train_texts), n_train, replace=False)
            idx_test = rng.choice(len(self.test_texts), n_test, replace=False)
            self.train_texts = [self.train_texts[i] for i in idx_train]
            self.train_labels = [self.train_labels[i] for i in idx_train]
            self.test_texts = [self.test_texts[i] for i in idx_test]
            self.test_labels = [self.test_labels[i] for i in idx_test]
            print(f"  ⚠️  Using limited size: Train={len(self.train_texts)}, Test={len(self.test_texts)} (test = train_limit × 0.15)")

        print(f"  ✅ IMDB ready! Train: {len(self.train_texts)}, Test: {len(self.test_texts)}")

    def _load_dbpedia(self):
        """Load DBpedia 14 dataset from Hugging Face (fancyzhx/dbpedia_14)."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("DBpedia requires datasets library. Install: pip install datasets")

        ds = load_dataset("fancyzhx/dbpedia_14")
        train_data = ds["train"]
        test_data = ds["test"]

        # DBpedia has 'title' and 'content' fields; combine them like AG News
        train_texts_combined = [f"{str(title)} {str(content)}" for title, content in zip(train_data["title"], train_data["content"])]
        test_texts_combined = [f"{str(title)} {str(content)}" for title, content in zip(test_data["title"], test_data["content"])]

        self.train_texts = train_texts_combined
        self.train_labels = list(train_data["label"])
        self.test_texts = test_texts_combined
        self.test_labels = list(test_data["label"])

        print(f"  📊 Full DBpedia Dataset: Train={len(self.train_texts)}, Test={len(self.test_texts)}")

        if self.dataset_size_limit is not None and self.dataset_size_limit > 0:
            rng = np.random.default_rng(42)
            n_train = min(self.dataset_size_limit, len(self.train_texts))
            n_test = min(int(self.dataset_size_limit * 0.15), len(self.test_texts))
            idx_train = rng.choice(len(self.train_texts), n_train, replace=False)
            idx_test = rng.choice(len(self.test_texts), n_test, replace=False)
            self.train_texts = [self.train_texts[i] for i in idx_train]
            self.train_labels = [self.train_labels[i] for i in idx_train]
            self.test_texts = [self.test_texts[i] for i in idx_test]
            self.test_labels = [self.test_labels[i] for i in idx_test]
            print(f"  ⚠️  Using limited size: Train={len(self.train_texts)}, Test={len(self.test_texts)} (test = train_limit × 0.15)")

        print(f"  ✅ DBpedia ready! Train: {len(self.train_texts)}, Test: {len(self.test_texts)}")

    def _load_yahoo_answers(self):
        """
        Load Yahoo Answers 10-category dataset.
        1. Read ``data/yahoo_answers/train.csv`` and ``data/yahoo_answers/test.csv`` if both exist.
        2. Otherwise download from Hugging Face and cache under ``data/yahoo_answers/``.
        """
        data_dir = os.path.join("data", "yahoo_answers")
        os.makedirs(data_dir, exist_ok=True)
        train_file = os.path.join(data_dir, "train.csv")
        test_file = os.path.join(data_dir, "test.csv")

        if os.path.exists(train_file) and os.path.exists(test_file):
            print(f"  ✅ Found local data files in {data_dir}/ directory. Loading...")
            train_df = pd.read_csv(train_file, header=None, names=['label', 'text'], quoting=1)
            test_df = pd.read_csv(test_file, header=None, names=['label', 'text'], quoting=1)
            self.train_texts = train_df['text'].fillna('').astype(str).tolist()
            self.train_labels = [(int(x) - 1) for x in train_df['label']]
            self.test_texts = test_df['text'].fillna('').astype(str).tolist()
            self.test_labels = [(int(x) - 1) for x in test_df['label']]
        else:
            try:
                from datasets import load_dataset
            except ImportError:
                raise ImportError("Yahoo Answers requires datasets library. Install: pip install datasets")

            print("  🌐 Local Yahoo Answers CSVs not both under data/yahoo_answers/. Downloading from Hugging Face...")
            ds = load_dataset("yassiracharki/Yahoo_Answers_10_categories_for_NLP")
            train_data = ds["train"]
            test_data = ds["test"]

            cols = train_data.column_names
            def _get_col(candidates):
                for c in candidates:
                    if c in cols:
                        return c
                return None
            label_col = _get_col(["class_index", "Class Index", "label"]) or cols[0]
            title_col = _get_col(["question_title", "Question Title"]) or cols[1]
            content_col = _get_col(["question_content", "Question Content"]) or cols[2]
            answer_col = _get_col(["best_answer", "Best Answer"]) or (cols[3] if len(cols) > 3 else None)

            def _combine_text(t, c, a):
                parts = [str(x or "").strip() for x in [t, c, a] if x is not None]
                return " ".join(p for p in parts if p) or " "

            if answer_col:
                train_texts = [_combine_text(t, c, a) for t, c, a in zip(train_data[title_col], train_data[content_col], train_data[answer_col])]
                test_texts = [_combine_text(t, c, a) for t, c, a in zip(test_data[title_col], test_data[content_col], test_data[answer_col])]
            else:
                train_texts = [_combine_text(t, c, None) for t, c in zip(train_data[title_col], train_data[content_col])]
                test_texts = [_combine_text(t, c, None) for t, c in zip(test_data[title_col], test_data[content_col])]
            train_labels_raw = list(train_data[label_col])
            test_labels_raw = list(test_data[label_col])

            self.train_texts = train_texts
            self.train_labels = [int(x) - 1 for x in train_labels_raw]
            self.test_texts = test_texts
            self.test_labels = [int(x) - 1 for x in test_labels_raw]

            train_save = pd.DataFrame({'label': [l + 1 for l in self.train_labels], 'text': self.train_texts})
            test_save = pd.DataFrame({'label': [l + 1 for l in self.test_labels], 'text': self.test_texts})
            train_save.to_csv(train_file, index=False, header=False, quoting=1)
            test_save.to_csv(test_file, index=False, header=False, quoting=1)
            print(f"  ✅ Saved to {data_dir}/ for future use.")

        print(f"  📊 Full Yahoo Answers Dataset: Train={len(self.train_texts)}, Test={len(self.test_texts)}")

        if self.dataset_size_limit is not None and self.dataset_size_limit > 0:
            rng = np.random.default_rng(42)
            n_train = min(self.dataset_size_limit, len(self.train_texts))
            n_test = min(int(self.dataset_size_limit * 0.15), len(self.test_texts))
            idx_train = rng.choice(len(self.train_texts), n_train, replace=False)
            idx_test = rng.choice(len(self.test_texts), n_test, replace=False)
            self.train_texts = [self.train_texts[i] for i in idx_train]
            self.train_labels = [self.train_labels[i] for i in idx_train]
            self.test_texts = [self.test_texts[i] for i in idx_test]
            self.test_labels = [self.test_labels[i] for i in idx_test]
            print(f"  ⚠️  Using limited size: Train={len(self.train_texts)}, Test={len(self.test_texts)} (test = train_limit × 0.15)")

        print(f"  ✅ Yahoo Answers ready! Train: {len(self.train_texts)}, Test: {len(self.test_texts)}")

    def _load_ag_news(self):
        """
        Load AG News from ``data/ag_news/train.csv`` and ``data/ag_news/test.csv``.

        Expected CSV format (no header): label, title, text (CharCNN / mhjabreel layout).
        Missing splits are downloaded from GitHub into ``data/ag_news/`` without overwriting
        any CSV that already exists locally.
        """
        data_dir = os.path.join("data", "ag_news")
        os.makedirs(data_dir, exist_ok=True)
        train_path = os.path.join(data_dir, "train.csv")
        test_path = os.path.join(data_dir, "test.csv")

        train_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
        test_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"

        try:
            has_train = os.path.exists(train_path)
            has_test = os.path.exists(test_path)

            if has_train and has_test:
                print(f"  ✅ Found local AG News files in {data_dir}/. Loading...")
                train_df = pd.read_csv(train_path, header=None, names=['label', 'title', 'text'])
                test_df = pd.read_csv(test_path, header=None, names=['label', 'title', 'text'])
            else:
                if not has_train and not has_test:
                    print("  🌐 No AG News CSVs under data/ag_news/. Downloading train + test from GitHub...")
                elif not has_train:
                    print("  🌐 Missing train.csv under data/ag_news/. Downloading train split only...")
                else:
                    print("  🌐 Missing test.csv under data/ag_news/. Downloading test split only...")

                if not has_train:
                    print(f"     Train source: {train_url}")
                    with urllib.request.urlopen(train_url, timeout=20) as response:
                        train_raw = response.read().decode('utf-8')
                    with open(train_path, 'w', encoding='utf-8') as f:
                        f.write(train_raw)
                if not has_test:
                    print(f"     Test source: {test_url}")
                    with urllib.request.urlopen(test_url, timeout=20) as response:
                        test_raw = response.read().decode('utf-8')
                    with open(test_path, 'w', encoding='utf-8') as f:
                        f.write(test_raw)

                train_df = pd.read_csv(train_path, header=None, names=['label', 'title', 'text'])
                test_df = pd.read_csv(test_path, header=None, names=['label', 'title', 'text'])
                print(f"  ✅ AG News CSVs ready under {data_dir}/.")

        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: Data loading failed: {e}")
            print("🛑 STRICT MODE: Synthetic data generation is DISABLED to ensure validity.")
            print(f"   Place train.csv and test.csv under {data_dir}/ or ensure network access for download.")
            raise e

        # Process Data
        # Combine title and text
        train_df['full_text'] = train_df['title'].astype(str) + ' ' + train_df['text'].astype(str)
        test_df['full_text'] = test_df['title'].astype(str) + ' ' + test_df['text'].astype(str)

        # Adjust labels 1-4 -> 0-3
        train_df['label'] = train_df['label'] - 1
        test_df['label'] = test_df['label'] - 1

        # Print full dataset size
        print(f"  📊 Full AG News Dataset: Train={len(train_df)}, Test={len(test_df)}")
        
        # Use full dataset by default
        # AG News full dataset: ~120,000 training samples, ~7,600 test samples
        # If dataset_size_limit is set, use it for faster experimentation (not recommended for paper reproduction)
        if hasattr(self, 'dataset_size_limit') and self.dataset_size_limit is not None:
            if self.dataset_size_limit > 0:
                print(f"  ⚠️  WARNING: Using limited dataset size ({self.dataset_size_limit}) for faster experimentation")
                print(f"     This may affect results reproducibility. For paper reproduction, use full dataset.")
                train_sample = train_df.sample(n=min(self.dataset_size_limit, len(train_df)), random_state=42)
                test_sample = test_df.sample(n=min(int(self.dataset_size_limit * 0.15), len(test_df)), random_state=42)
            else:
                # Use full dataset
                train_sample = train_df
                test_sample = test_df
        else:
            # Use full dataset (default, per paper)
            train_sample = train_df
            test_sample = test_df

        self.train_texts = train_sample['full_text'].tolist()
        self.train_labels = train_sample['label'].tolist()
        self.test_texts = test_sample['full_text'].tolist()
        self.test_labels = test_sample['label'].tolist()

        print(f"  ✅ Dataset ready! Train: {len(self.train_texts)}, Test: {len(self.test_texts)}")
        if len(self.train_texts) < len(train_df) or len(self.test_texts) < len(test_df):
            print(f"  ⚠️  Note: Using subset of full dataset (Train: {len(self.train_texts)}/{len(train_df)}, "
                  f"Test: {len(self.test_texts)}/{len(test_df)})")
        else:
            print(f"  ✅ Using FULL AG News dataset (per paper requirements)")

    def get_empty_loader(self) -> DataLoader:
        """Return an empty loader for data-agnostic attackers."""
        return DataLoader(NewsDataset([], [], self.tokenizer, max_length=self.max_length), batch_size=self.batch_size, shuffle=False)

    def get_proxy_eval_loader(self, sample_size: int = 128) -> DataLoader:
        """
        Small clean proxy set for attacker-side global-loss estimation
        (used by ALIE, attack/alie.py).

        Fairness (critical): the proxy is drawn from the TRAINING distribution
        (``self.train_texts``), NEVER from the test set. The attacker must not
        be able to optimise its malicious direction against the exact samples
        used to report clean accuracy / PPL, so the proxy pool is disjoint from
        the evaluation set by construction. This matches the standard FL threat
        model where a participant holds a small clean dataset (cf. FLTrust's
        server root set) and keeps the reported metrics honest.

        The selection is deterministic (dedicated seed) and class-stratified for
        a balanced loss estimate. Only attacker clients call this method, and it
        does not touch ``train_texts`` or the client partition, so existing
        (Hallucination / fedavg / baseline) experiments are unaffected.
        """
        if not self.train_texts:
            return self.get_empty_loader()
        # Dedicated seed offset so the proxy selection is independent of the
        # test-sampling / partition RNGs and reproducible across runs.
        rng = np.random.default_rng(int(self.test_seed) + 90007)
        n_train = len(self.train_texts)
        target = min(int(sample_size), n_train)
        labels_arr = np.asarray(self.train_labels)
        classes = np.unique(labels_arr)
        # Class-stratified quota (balanced proxy -> less biased F(w'_g) estimate).
        per_class = max(1, target // max(1, len(classes)))
        chosen: List[int] = []
        for c in classes:
            c_idx = np.where(labels_arr == c)[0]
            take = min(per_class, len(c_idx))
            if take > 0:
                chosen.extend(rng.choice(c_idx, size=take, replace=False).tolist())
        # Top up to `target` from the remaining pool if stratified quota fell short.
        if len(chosen) < target:
            remaining = np.setdiff1d(np.arange(n_train), np.asarray(chosen, dtype=int))
            if len(remaining) > 0:
                extra = rng.choice(
                    remaining, size=min(target - len(chosen), len(remaining)), replace=False
                )
                chosen.extend(extra.tolist())
        chosen = sorted(chosen)
        proxy_texts = [self.train_texts[i] for i in chosen]
        proxy_labels = [self.train_labels[i] for i in chosen]
        dataset = NewsDataset(proxy_texts, proxy_labels, self.tokenizer, max_length=self.max_length)
        return DataLoader(dataset, batch_size=self.test_batch_size, shuffle=False)

    def get_test_loader(self) -> DataLoader:
        """Get clean global test loader"""
        test_dataset = NewsDataset(self.test_texts, self.test_labels, self.tokenizer, max_length=self.max_length)
        return DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False)

