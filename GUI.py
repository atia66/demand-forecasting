import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from datetime import date

from Model import RNNModel


class DemandForecastGUI:

    TABLE_COLS = [
        ("#",             "#"),
        ("Store",         "Store ID"),
        ("Product",       "Product ID"),
        ("Category",      "Category"),
        ("Region",        "Region"),
        ("Price",         "Price"),
        ("Discount",      "Discount"),
        ("Inventory",     "Inventory Level"),
        ("Units Sold",    "Units Sold"),
        ("Units Ordered", "Units Ordered"),
        ("Weather",       "Weather Condition"),
        ("Season",        "Seasonality"),
        ("Promo",         "Promotion"),
        ("Epidemic",      "Epidemic"),
        ("Comp. Price",   "Competitor Pricing"),
    ]

    def __init__(self):
        self.load_model()
        self.history     = []
        self.history_raw = []

        self.root = tk.Tk()
        self.root.title("Demand Forecasting System - RNN")
        self.root.geometry("1150x980")
        self.root.configure(bg="#1a1a2e")

        self.setup_ui()

    def load_model(self):
        try:
            with open("model_artifacts.pkl", "rb") as f:
                artifacts = pickle.load(f)

            self.feature_scaler     = artifacts["feature_scaler"]
            self.target_scaler      = artifacts["target_scaler"]
            self.encoders           = artifacts["encoders"]
            self.feature_cols       = artifacts["feature_cols"]
            self.numerical_features = artifacts["numerical_features"]
            self.valid_categories   = artifacts["valid_categories"]

            checkpoint_weights = torch.load("demand_model.pth", map_location="cpu", weights_only=True)
            checkpoint_embeddings = torch.load("embedding.pth", map_location="cpu", weights_only=True)
            self.embeddings = {}
            for col, weight in checkpoint_embeddings["embeddings"].items():
                emb = nn.Embedding.from_pretrained(weight, freeze=True)
                self.embeddings[col] = emb
            input_size = self._calculate_input_size()
            self.model = RNNModel(input_size)
            self.model.load_state_dict(checkpoint_weights["model_state_dict"])
            self.model.eval()

            self._key_alias = {k.lower().strip(): k for k in self.valid_categories}

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model:\n{e}")
            exit()

    def _calculate_input_size(self):
        size = 0
        for col in self.feature_cols:
            if col in self.valid_categories:
                size += self.embeddings[col].embedding_dim
            else:
                size += 1
        return size

    def _resolve_key(self, *candidates):
        
        for c in candidates:
            if c in self.valid_categories:
                return c
        for c in candidates:
            real = self._key_alias.get(c.lower().strip())
            if real:
                return real
        for c in candidates:
            cl = c.lower().strip()
            for k_low, k_real in self._key_alias.items():
                if cl in k_low or k_low in cl:
                    return k_real
        print(f"Could not resolve key from {candidates}. "
              f"Available: {list(self.valid_categories.keys())}")
        return None

    def _make_combo(self, parent, preferred_key, row, col,
                    label_text, label_col, *alt_keys):
        real_key = self._resolve_key(preferred_key, *alt_keys)
        values   = self.valid_categories.get(real_key, []) if real_key else []

        if not values:
            print(f"WARNING: no values for {preferred_key!r} "
                  f"(resolved={real_key!r}). "
                  f"Available: {list(self.valid_categories.keys())}")

        tk.Label(parent, text=label_text, bg="#16213e", fg="#e0e0e0",
                 font=("Segoe UI", 10)).grid(
                     row=row, column=label_col, padx=10, pady=7, sticky="w")
        combo = ttk.Combobox(parent, values=values, width=18, state="readonly")
        if values:
            combo.set(values[0])
        combo.grid(row=row, column=col, padx=10, pady=7)
        combo._real_key = real_key
        return combo

    def _make_toggle(self, parent, var, row, col):
        frame = tk.Frame(parent, bg="#16213e")
        frame.grid(row=row, column=col, padx=10, pady=7, sticky="w")
        btn = tk.Button(frame, font=("Segoe UI", 10, "bold"),
                        width=8, relief=tk.RAISED, bd=2, cursor="hand2")

        def _refresh(*_):
            if var.get():
                btn.config(text="Yes", bg="#00c853", fg="white",
                           activebackground="#69f0ae")
            else:
                btn.config(text="No", bg="#c62828", fg="white",
                           activebackground="#ef9a9a")

        btn.config(command=lambda: [var.set(0 if var.get() else 1), _refresh()])
        _refresh()
        btn.pack()

    def _entry_row(self, parent, row, l0, e0, attr0, l1, e1, attr1):
        for col, label, default, attr in [(0, l0, e0, attr0), (2, l1, e1, attr1)]:
            tk.Label(parent, text=label, bg="#16213e", fg="#e0e0e0",
                     font=("Segoe UI", 10)).grid(
                         row=row, column=col, padx=10, pady=7, sticky="w")
            w = tk.Entry(parent, width=20, font=("Segoe UI", 10))
            w.insert(0, default)
            w.grid(row=row, column=col + 1, padx=10, pady=7)
            setattr(self, attr, w)

    def setup_ui(self):
        tk.Label(self.root, text="Demand Forecasting System",
                 font=("Segoe UI", 22, "bold"),
                 bg="#1a1a2e", fg="#ffd700").pack(pady=12)

        outer  = tk.Frame(self.root, bg="#16213e", relief=tk.RAISED, bd=3)
        outer.pack(padx=20, pady=5, fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(outer, bg="#16213e", highlightthickness=0)
        vsb    = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        main = tk.Frame(canvas, bg="#16213e")
        wid  = canvas.create_window((0, 0), window=main, anchor="nw")

        main.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(wid, width=e.width))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        # ── Input section ─────────────────────────────────────────────────────
        inf = tk.LabelFrame(main, text="Current Day Input",
                            font=("Segoe UI", 12, "bold"),
                            bg="#16213e", fg="#00d4ff",
                            relief=tk.GROOVE, bd=2)
        inf.pack(padx=15, pady=10, fill=tk.X)

        self.store_entry    = self._make_combo(inf, "Store ID",   0, 1, "Store ID:",   0)
        self.product_entry  = self._make_combo(inf, "Product ID", 0, 3, "Product ID:", 2)
        self.category_entry = self._make_combo(inf, "Category",   1, 1, "Category:",   0)
        self.region_entry   = self._make_combo(inf, "Region",     1, 3, "Region:",     2)

        self._entry_row(inf, row=2,
                        l0="Price:",    e0="80.16", attr0="price_entry",
                        l1="Discount:", e1="0.15",  attr1="discount_entry")
        self._entry_row(inf, row=3,
                        l0="Inventory Level:", e0="117", attr0="inventory_entry",
                        l1="Units Sold:",      e1="50",  attr1="units_sold_entry")
        self._entry_row(inf, row=4,
                        l0="Units Ordered:",      e0="60",    attr0="units_ordered_entry",
                        l1="Competitor Pricing:", e1="92.02", attr1="comp_price_entry")

        self.weather_entry = self._make_combo(
            inf, "Weather Condition", 5, 1, "Weather Condition:", 0,
            "Weather", "WeatherCondition", "weather_condition")
        self.season_entry = self._make_combo(
            inf, "Seasonality", 5, 3, "Seasonality:", 2,
            "Season", "SeasonType", "season_type")

        tk.Label(inf, text="Promotion:", bg="#16213e", fg="#e0e0e0",
                 font=("Segoe UI", 10)).grid(row=6, column=0, padx=10, pady=7, sticky="w")
        self.promotion_var = tk.IntVar(value=1)
        self._make_toggle(inf, self.promotion_var, row=6, col=1)

        tk.Label(inf, text="Epidemic:", bg="#16213e", fg="#e0e0e0",
                 font=("Segoe UI", 10)).grid(row=6, column=2, padx=10, pady=7, sticky="w")
        self.epidemic_var = tk.IntVar(value=0)
        self._make_toggle(inf, self.epidemic_var, row=6, col=3)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_frame = tk.Frame(main, bg="#16213e")
        btn_frame.pack(pady=8)
        for label, bg, fg, cmd in [
            ("Add Day",  "#00d4ff", "#1a1a2e", self.add_to_history),
            ("Forecast", "#ffd700", "#1a1a2e", self.predict_demand),
            ("Load CSV", "#9b59b6", "white",   self.load_from_csv),
            ("Clear",    "#e74c3c", "white",   self.clear_history),
        ]:
            tk.Button(btn_frame, text=label, font=("Segoe UI", 11, "bold"),
                      bg=bg, fg=fg, relief=tk.RAISED, bd=3, width=12,
                      command=cmd, cursor="hand2").pack(side=tk.LEFT, padx=5)

        # ── History table ─────────────────────────────────────────────────────
        hf = tk.LabelFrame(main, text="Historical Sequence  (last 14 days)",
                           font=("Segoe UI", 12, "bold"),
                           bg="#16213e", fg="#00d4ff",
                           relief=tk.GROOVE, bd=2)
        hf.pack(padx=15, pady=6, fill=tk.X)

        self.history_label = tk.Label(
            hf, text="0 / 14 days added",
            font=("Segoe UI", 10, "bold"), bg="#16213e", fg="#ff9500")
        self.history_label.pack(anchor="w", padx=8, pady=(4, 0))

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("History.Treeview",
                        background="#0f3460", foreground="#e0e0e0",
                        fieldbackground="#0f3460", rowheight=24,
                        font=("Segoe UI", 9))
        style.configure("History.Treeview.Heading",
                        background="#1a1a2e", foreground="#ffd700",
                        font=("Segoe UI", 9, "bold"), relief="flat")
        style.map("History.Treeview",
                  background=[("selected", "#00d4ff")],
                  foreground=[("selected", "#1a1a2e")])

        col_ids = [c[0] for c in self.TABLE_COLS]
        self.tree = ttk.Treeview(hf, columns=col_ids, show="headings",
                                 height=7, style="History.Treeview")

        widths = {
            "#": 28, "Store": 65, "Product": 70, "Category": 80,
            "Region": 70, "Price": 58, "Discount": 62, "Inventory": 75,
            "Units Sold": 72, "Units Ordered": 88, "Weather": 90,
            "Season": 80, "Promo": 50, "Epidemic": 62, "Comp. Price": 82,
        }
        for cid in col_ids:
            self.tree.heading(cid, text=cid)
            self.tree.column(cid, width=widths.get(cid, 70),
                             anchor="center", stretch=True)

        self.tree.tag_configure("odd",    background="#0f3460")
        self.tree.tag_configure("even",   background="#16213e")
        self.tree.tag_configure("latest", background="#1a4a1a")

        hsb = ttk.Scrollbar(hf, orient="horizontal", command=self.tree.xview)
        self.tree.configure(xscrollcommand=hsb.set)
        self.tree.pack(padx=8, pady=4, fill=tk.X)
        hsb.pack(padx=8, fill=tk.X)

        # ── Result section ────────────────────────────────────────────────────
        rf = tk.LabelFrame(main, text="Forecast Result",
                           font=("Segoe UI", 12, "bold"),
                           bg="#16213e", fg="#00d4ff",
                           relief=tk.GROOVE, bd=2)
        rf.pack(padx=15, pady=10, fill=tk.X)

        self.result_label = tk.Label(rf, text="Predicted Demand: ---",
                                     font=("Segoe UI", 20, "bold"),
                                     bg="#16213e", fg="#00ff88")
        self.result_label.pack(pady=12)

        self.details_label = tk.Label(
            rf, text="Add 14 consecutive days to enable forecasting",
            font=("Segoe UI", 10), bg="#16213e", fg="#b0b0b0",
            justify=tk.LEFT)
        self.details_label.pack(pady=8)

    def _encode(self, widget):
        real_key = widget._real_key
        if real_key is None:
            raise ValueError(
                f"Unresolved key. Available: {list(self.valid_categories.keys())}")
        if real_key not in self.valid_categories:
            raise ValueError(
                f"{real_key!r} is not a categorical feature. "
                f"Available: {list(self.valid_categories.keys())}")
        val   = widget.get()
        valid = self.valid_categories[real_key]
        if val not in valid:
            raise ValueError(
                f"Invalid value for {real_key!r}: {val!r}\n"
                f"Valid options: {valid}")
        return self.encoders[real_key].transform([val])[0]

    def get_current_input(self):
        raw = {
            "Store ID":           self.store_entry.get(),
            "Product ID":         self.product_entry.get(),
            "Category":           self.category_entry.get(),
            "Region":             self.region_entry.get(),
            "Price":              float(self.price_entry.get()),
            "Discount":           float(self.discount_entry.get()),
            "Inventory Level":    int(self.inventory_entry.get()),
            "Units Sold":         int(self.units_sold_entry.get()),
            "Units Ordered":      int(self.units_ordered_entry.get()),
            "Competitor Pricing": float(self.comp_price_entry.get()),
            "Weather Condition":  self.weather_entry.get(),
            "Seasonality":        self.season_entry.get(),
            "Promotion":          "Yes" if self.promotion_var.get() else "No",
            "Epidemic":           "Yes" if self.epidemic_var.get() else "No",
        }

        # Date as ordinal int — same conversion used in training
        feature_dict = {
            "Date":               date.today().toordinal(),
            "Store ID":           self._encode(self.store_entry),
            "Product ID":         self._encode(self.product_entry),
            "Category":           self._encode(self.category_entry),
            "Region":             self._encode(self.region_entry),
            "Inventory Level":    int(self.inventory_entry.get()),
            "Units Sold":         int(self.units_sold_entry.get()),
            "Units Ordered":      int(self.units_ordered_entry.get()),
            "Price":              float(self.price_entry.get()),
            "Discount":           float(self.discount_entry.get()),
            "Weather Condition":  self._encode(self.weather_entry),
            "Promotion":          self.promotion_var.get(),
            "Competitor Pricing": float(self.comp_price_entry.get()),
            "Seasonality":        self._encode(self.season_entry),
            "Epidemic":           self.epidemic_var.get(),
        }

        row_data = []
        for col in self.feature_cols:
            if col in feature_dict:
                row_data.append(feature_dict[col])
            else:
                col_l = col.lower()
                match = next((v for k, v in feature_dict.items()
                              if k.lower() == col_l), 0)
                row_data.append(match)

        row_arr = np.array([row_data], dtype=np.float64)

        # Scale all numerical features (Date included)
        if self.feature_scaler and self.numerical_features:
            num_indices = [self.feature_cols.index(col)
                           for col in self.numerical_features
                           if col in self.feature_cols]
            row_arr[:, num_indices] = self.feature_scaler.transform(
                row_arr[:, num_indices])

        processed_features = []
        for i, col in enumerate(self.feature_cols):
            if col in self.valid_categories:
                cat_val  = torch.tensor([int(row_arr[0, i])], dtype=torch.long)
                embedded = self.embeddings[col](cat_val)
                processed_features.append(embedded)
            else:
                num_val = torch.tensor([[row_arr[0, i]]], dtype=torch.float32)
                processed_features.append(num_val)

        feature_tensor = torch.cat(processed_features, dim=1).squeeze(0)
        return feature_tensor, raw

    def _refresh_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        col_keys = [c[1] for c in self.TABLE_COLS]

        for i, raw in enumerate(self.history_raw):
            vals = []
            for key in col_keys:
                if key == "#":
                    vals.append(str(i + 1))
                else:
                    v = raw.get(key, "")
                    if isinstance(v, float):
                        v = f"{v:.2f}"
                    vals.append(str(v))

            tag = ("latest" if i == len(self.history_raw) - 1
                   else ("even" if i % 2 == 0 else "odd"))
            self.tree.insert("", "end", values=vals, tags=(tag,))

        n = len(self.history_raw)
        self.history_label.config(
            text=f"{n} / 14 days added",
            fg="#00ff88" if n == 14 else "#ff9500")

    def add_to_history(self):
        try:
            feature_tensor, raw = self.get_current_input()
            self.history.append(feature_tensor.detach())
            self.history_raw.append(raw)

            if len(self.history) > 14:
                self.history.pop(0)
                self.history_raw.pop(0)

            self._refresh_table()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add:\n{e}")

    def clear_history(self):
        self.history     = []
        self.history_raw = []
        self._refresh_table()
        self.result_label.config(text="Predicted Demand: ---")
        self.details_label.config(
            text="Add 14 consecutive days to enable forecasting")
        messagebox.showinfo("Cleared", "History reset successfully!")

    def load_from_csv(self):
        try:
            path = filedialog.askopenfilename(
                title="Select CSV File",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
            if not path:
                return

            df_raw = pd.read_csv(path)
            df     = df_raw.copy()

            # ── Filter by raw string values BEFORE encoding ───────────────
            prod_raw  = self.product_entry.get()
            store_raw = self.store_entry.get()
            mask      = ((df_raw["Product ID"].astype(str) == str(prod_raw)) &
                        (df_raw["Store ID"].astype(str)   == str(store_raw)))
            
            df_raw = df_raw[mask].copy()
            df     = df_raw.copy()

            if len(df_raw) == 0:
                messagebox.showwarning(
                    "No Match",
                    f"No rows found for:\n"
                    f"Product: {prod_raw}\nStore: {store_raw}\n\n"
                    f"Check that the CSV contains these exact values.")
                return

            # Encode categorical columns
            for col in df.select_dtypes(include=["object"]).columns:
                if col in self.valid_categories:
                    df[col] = self.encoders[col].transform(df[col].astype(str))

            # Date: convert to ordinal
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"]).map(lambda d: d.toordinal())

            subset     = df.sort_values("Date").tail(14)
            subset_raw = df_raw.loc[subset.index]

            if len(subset) < 14:
                messagebox.showwarning(
                    "Insufficient Data",
                    f"Only {len(subset)} days found for this store/product. Need 14.")
                return


            self.history = []
            for _, row in subset.iterrows():
                row_data = row[self.feature_cols].values.reshape(1, -1).astype(np.float64)

                if self.feature_scaler and self.numerical_features:
                    num_indices = [self.feature_cols.index(col)
                                   for col in self.numerical_features
                                   if col in self.feature_cols]
                    row_data[:, num_indices] = self.feature_scaler.transform(
                        row_data[:, num_indices])

                processed_features = []
                for i, col in enumerate(self.feature_cols):
                    if col in self.valid_categories:
                        cat_val  = torch.tensor([int(row_data[0, i])], dtype=torch.long)
                        embedded = self.embeddings[col](cat_val)
                        processed_features.append(embedded)
                    else:
                        num_val = torch.tensor([[row_data[0, i]]], dtype=torch.float32)
                        processed_features.append(num_val)

                feature_tensor = torch.cat(processed_features, dim=1).squeeze(0)
                self.history.append(feature_tensor.detach())

            self.history_raw = []
            for _, r in subset_raw.iterrows():
                self.history_raw.append({
                    "Store ID":           str(r.get("Store ID", "")),
                    "Product ID":         str(r.get("Product ID", "")),
                    "Category":           str(r.get("Category", "")),
                    "Region":             str(r.get("Region", "")),
                    "Price":              r.get("Price", 0),
                    "Discount":           r.get("Discount", 0),
                    "Inventory Level":    r.get("Inventory Level", 0),
                    "Units Sold":         r.get("Units Sold", 0),
                    "Units Ordered":      r.get("Units Ordered", 0),
                    "Competitor Pricing": r.get("Competitor Pricing", 0),
                    "Weather Condition":  str(r.get("Weather Condition", "")),
                    "Seasonality":        str(r.get("Seasonality", "")),
                    "Promotion":          "Yes" if r.get("Promotion", 0) else "No",
                    "Epidemic":           "Yes" if r.get("Epidemic", 0) else "No",
                })

            self._refresh_table()
            messagebox.showinfo(
                "Success",
                f"Loaded 14 days!\n"
                f"{self.product_entry.get()} @ {self.store_entry.get()}")

        except Exception as e:
            messagebox.showerror("Error", f"CSV load failed:\n{e}")

    def predict_demand(self):
        if len(self.history) < 14:
            messagebox.showwarning(
                "Insufficient Data",
                f"Need 14 days. Currently have: {len(self.history)}")
            return
        try:
            X_seq = torch.stack(self.history[-14:]).unsqueeze(0)  # (1, 14, feat_dim)

            with torch.no_grad():
                pred_scaled = self.model(X_seq).squeeze().item()

            pred = max(0, int(
                self.target_scaler.inverse_transform([[pred_scaled]])[0][0]))

            self.result_label.config(text=f"Predicted Demand: {pred} units")
            self.details_label.config(text=(
                f"Based on 14-day sequence\n"
                f"Store: {self.store_entry.get()}  |  "
                f"Product: {self.product_entry.get()}\n"
                f"Category: {self.category_entry.get()}  |  "
                f"Region: {self.region_entry.get()}"
            ))
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed:\n{e}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = DemandForecastGUI()
    app.run()