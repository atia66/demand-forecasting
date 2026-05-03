import torch
import torch.nn as nn
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from data_processing import DemandDataset
from Model import RNNModel
import matplotlib.pyplot as plt
df = pd.read_csv("./demand_forecasting.csv")

if __name__ == "__main__":
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)

    train_dataset = DemandDataset(train_df, "Demand")
    val_dataset   = DemandDataset(val_df, "Demand", artifacts=train_dataset.get_artifacts())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
    val_df_raw = val_df.copy()
    for (_, _), group in val_df_raw.groupby(["Store ID", "Product ID"]):
        if len(group) >= 15:
            group_sorted = group.sort_values("Date")
            group_sorted.iloc[:14].to_csv("val_sequence.csv", index=False)   # 14 input days
            break

    input_size = train_dataset.get_feature_size()

    model     = RNNModel(input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs        = 50
    train_losses, val_losses = [], []
    print("🚀 Starting training...\n")
    best_val_loss = float("inf")
    counter = 0
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            
            optimizer.zero_grad()
            preds = model(X).squeeze()
            loss  = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)  
        train_losses.append(train_loss)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                preds = model(X).squeeze()
                loss  = criterion(preds, y)
                val_loss += loss.item()
            val_loss /= len(val_loader)  
            val_losses.append(val_loss)
        print(f"Epoch {epoch+1:2d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
            }, "demand_model.pth")
            print("\n✅ New best validation loss found")
            counter = 0
        else:
            counter += 1
            if counter >= 7:
                print("\n⏳ Early stopping triggered")
                break

    torch.save({
        'embeddings':{k: emb.weight.data for k, emb in train_dataset.embeddings.items()},
    }, "embedding.pth")
    
    artifacts = train_dataset.get_artifacts()
    with open("model_artifacts.pkl", "wb") as f:
        pickle.dump({
            "feature_scaler":     artifacts["feature_scaler"],
            "target_scaler":      artifacts["target_scaler"],
            "encoders":           artifacts["encoders"],
            "feature_cols":       artifacts["feature_cols"],
            "numerical_features": artifacts["numerical_features"],
            "valid_categories":   artifacts["valid_categories"],
        }, f)
    
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Normalized Loss")
    plt.legend()

    plt.savefig("loss_plot.png") 
    plt.show()                    
    
    print("\n💾 Model saved to 'demand_model.pth'")
    
    print("💾 Artifacts saved to 'model_artifacts.pkl'")