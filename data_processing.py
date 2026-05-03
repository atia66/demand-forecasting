import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DemandDataset(Dataset):

    def __init__(self, df, target_col, seq_len=14, artifacts=None):
        self.seq_len    = seq_len
        self.df         = df.copy()
        self.target_col = target_col
        self.X          = []
        self.y          = []

        if artifacts is None: # Training mode
            self.encoding()
            self.scale_features()
            self.set_embeddings()
        else: # Inference mode — reuse training artifacts
            self.encoders           = artifacts["encoders"]
            self.valid_categories   = artifacts["valid_categories"]
            self.feature_cols       = artifacts["feature_cols"]
            self.numerical_features = artifacts["numerical_features"]
            self.feature_scaler     = artifacts["feature_scaler"]
            self.target_scaler      = artifacts["target_scaler"]
            self.embeddings         = artifacts["embeddings"]
            self._apply_encoders()
            self._apply_scalers()

        self.process_data()

    def encoding(self):
        self.encoders         = {}
        self.valid_categories = {}
        self.feature_cols     = [c for c in self.df.columns if c != self.target_col]

        categorical_cols = self.df.select_dtypes(include=["object"]).columns.tolist()

        for col in categorical_cols:
            if col == "Date":
                self.df[col] = pd.to_datetime(self.df[col]).map(lambda d: d.toordinal()) # get timestamp from date 
            else:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])
                self.encoders[col] = le
                self.valid_categories[col] = list(le.classes_)

        self.numerical_features = [
            col for col in self.feature_cols
            if col not in self.valid_categories
        ]

    def scale_features(self):
        self.feature_scaler = StandardScaler()
        self.df[self.numerical_features] = self.feature_scaler.fit_transform(
            self.df[self.numerical_features])

        self.target_scaler = StandardScaler()
        self.df[[self.target_col]] = self.target_scaler.fit_transform(
            self.df[[self.target_col]])

    def set_embeddings(self):
        self.embeddings = {}
        for col in self.valid_categories:
            num_classes = len(self.valid_categories[col])
            self.embeddings[col] = nn.Embedding(num_classes, 5)

    def _apply_encoders(self):
        for col in self.df.select_dtypes(include=["object"]).columns:
            if col == "Date":
                # Same ordinal conversion as training — works on any date
                self.df[col] = pd.to_datetime(self.df[col]).map(lambda d: d.toordinal())
            elif col in self.encoders:
                self.df[col] = self.encoders[col].transform(self.df[col].astype(str))

    def _apply_scalers(self):
        self.df[self.numerical_features] = self.feature_scaler.transform(
            self.df[self.numerical_features])
        self.df[[self.target_col]] = self.target_scaler.transform(
            self.df[[self.target_col]])

    def process_data(self):
        groups = self.df.groupby(["Product ID", "Store ID"])

        for (_, _), group in groups:
            group = group.sort_values("Date")
            data  = group[self.feature_cols]

            processed_data = []
            for col in data.columns:
                if col in self.valid_categories:
                    cat_data = torch.tensor(data[col].values, dtype=torch.long)
                    embedded = self.embeddings[col](cat_data)
                    processed_data.append(embedded.detach())
                else:
                    num_data = torch.tensor(
                        data[col].values, dtype=torch.float32).unsqueeze(1)
                    processed_data.append(num_data)

            data   = torch.cat(processed_data, dim=1)
            target = group[self.target_col].values

            for i in range(len(group) - self.seq_len):
                self.X.append(data[i:i + self.seq_len].detach())
                self.y.append(target[i + self.seq_len])

        self.X = torch.stack(self.X)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32)

    def get_artifacts(self):
        return {
            "encoders":           self.encoders,
            "valid_categories":   self.valid_categories,
            "feature_cols":       self.feature_cols,
            "numerical_features": self.numerical_features,
            "feature_scaler":     self.feature_scaler,
            "target_scaler":      self.target_scaler,
            "embeddings":         self.embeddings,
        }

    def get_feature_size(self):
        return self.X.shape[-1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
