import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import psutil
from datetime import datetime


def print_progress(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    tqdm.write(f"[{timestamp}] 🚧 {message}")


def format_file_size(path):
    size = path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} GB"


def generate_tcga_synthetic_data(output_dir="data"):
    np.random.seed(42)
    output_path = Path(output_dir)

    print_progress(f"Available memory: {psutil.virtual_memory().available / 1e9:.1f} GB")

    with tqdm(total=7, desc="🚀 Overall Progress") as main_pbar:
        print_progress("Initializing synthetic data generation")
        print(f"📂 Output directory: {output_path.resolve()}")
        main_pbar.update(1)

        # Class distribution
        class_distribution = {
            "LumA": 0.476, "LumB": 0.184, "Basal": 0.157,
            "HER2E": 0.069, "Normal": 0.114
        }

        # Reduced parameters
        n_samples = 200
        feature_sizes = {'methyl': 5000, 'mirna': 500, 'rna_exp': 500}

        print_progress("Generating class labels...")
        labels = np.random.choice(
            list(class_distribution.keys()),
            size=n_samples,
            p=list(class_distribution.values())
        )
        main_pbar.update(1)

        def simulate_block(n_feat, name):
            """Optimized data generation with progress tracking"""
            with tqdm(total=4, desc=f"🧬 Generating {name}", leave=False) as pbar:
                X = np.empty((n_samples, n_feat))
                pbar.update(1)

                # Reduced chunk size for small features
                chunk_size = min(200, n_feat)
                for i in range(0, n_feat, chunk_size):
                    end = min(i + chunk_size, n_feat)
                    chunk, _ = make_classification(
                        n_samples=n_samples,
                        n_features=end - i,
                        n_informative=int(0.6 * (end - i)),
                        n_redundant=int(0.1 * (end - i)),
                        n_classes=5,
                        weights=list(class_distribution.values()),
                        random_state=42 + i
                    )

                    if name == "methyl":
                        chunk = MinMaxScaler(feature_range=(0, 1)).fit_transform(chunk)
                    else:
                        chunk = MinMaxScaler(feature_range=(-10, 20)).fit_transform(chunk)

                    X[:, i:end] = chunk
                    pbar.update(0.5)

                pbar.update(3.5 - (n_feat / chunk_size) * 0.5)
                return np.round(X, 4)

        print_progress("Generating omics data...")
        methyl = simulate_block(5000, "Methylation")
        mirna = simulate_block(500, "miRNA")
        rna_exp = simulate_block(500, "RNA Expression")
        main_pbar.update(1)

        print_progress("Creating DataFrames...")
        sample_ids = [f"TCGA-{i:04d}" for i in range(n_samples)]

        # Methylation data (5000 features)
        methyl_df = pd.DataFrame(
            methyl,
            index=sample_ids,
            columns=[f"cg{i:04d}" for i in range(5000)]
        )

        # miRNA data (500 features)
        mirna_df = pd.DataFrame(
            mirna, index=sample_ids,
            columns=[f"hsa-miR-{i:03d}" for i in range(500)]
        )

        # RNA expression data (500 features)
        rna_exp_df = pd.DataFrame(
            rna_exp, index=sample_ids,
            columns=[f"ENSG{i:06d}" for i in range(500)]
        )

        clinical_df = pd.DataFrame({
            "sample_id": sample_ids,
            "age": np.random.normal(58, 12, n_samples).astype(int),
            "stage": np.random.choice(["I", "II", "III"], size=n_samples, p=[0.3, 0.5, 0.2]),
            "subtype": labels
        })
        main_pbar.update(1)

        output_path.mkdir(parents=True, exist_ok=True)

        def save_with_progress(df, filename):
            with tqdm(total=1, desc=f"💾 Saving {filename}", leave=False) as pbar:
                path = output_path / filename
                df.to_csv(path, float_format="%.4f")
                pbar.set_postfix(size=format_file_size(path))
                pbar.update(1)

        print_progress("Saving data files...")
        with tqdm(total=4, desc="📁 Writing Files") as pbar:
            save_with_progress(methyl_df, "methyl.csv")
            pbar.update(1)
            save_with_progress(mirna_df, "mirna.csv")
            pbar.update(1)
            save_with_progress(rna_exp_df, "rna_exp.csv")
            pbar.update(1)
            save_with_progress(clinical_df, "clinical.csv")
            pbar.update(1)

        main_pbar.update(1)


def generate_data_config(output_dir="data"):
    with tqdm(total=1, desc="⚙️  Generating Config") as pbar:
        output_path = Path(output_dir)
        config = {
            "data_dir": str(output_path.resolve()),
            "omics": {
                "methyl": {
                    "file": "methyl.csv",
                    "dtype": "float32",
                    "reshape": [50, 100]  # 50x100 = 5000
                },
                "mirna": {"file": "mirna.csv", "dtype": "float32"},
                "rna_exp": {"file": "rna_exp.csv", "dtype": "float32"}
            },
            "labels": {
                "source": "clinical.csv",
                "column": "subtype",
                "class_map": {
                    "LumA": 0, "LumB": 1, "Basal": 2,
                    "HER2E": 3, "Normal": 4
                }
            }
        }

        config_path = output_path / "data_sim_config.yaml"
        with config_path.open('w') as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=False)

        pbar.update(1)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic multi-omics dataset")
    parser.add_argument("--output_dir", default="./data", help="Output directory")
    args = parser.parse_args()

    print("🔬 Starting synthetic data generation")
    try:
        with tqdm(total=2, desc="🌈 Main Process") as main_pbar:
            generate_tcga_synthetic_data(args.output_dir)
            main_pbar.update(1)
            generate_data_config(args.output_dir)
            main_pbar.update(1)

        print("\n🎉 Success! Generated files:")
        for f in Path(args.output_dir).glob("*"):
            print(f" - {f.name} ({format_file_size(f)})")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()