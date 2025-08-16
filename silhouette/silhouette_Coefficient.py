import os
import re
import time
import argparse
import multiprocessing

import numpy as np
import pandas as pd
import jieba

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import pairwise_distances

# -------------------------- Optional Dependencies/Environment Detection --------------------------
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

def get_progress_bar(iterable=None, total=None, desc="Processing"):
    if TQDM_AVAILABLE:
        return tqdm(iterable, total=total, desc=desc)
    class SimpleProgressBar:
        def __init__(self, total, desc="Processing"):
            self.total = total or 0
            self.desc = desc
            self.n = 0
        def update(self, n=1):
            self.n += n
            if self.total:
                percent = int(100 * self.n / self.total)
                print(f"{self.desc}: {percent}%", end="\r")
        def close(self):
            if self.total:
                print(f"{self.desc}: 100%")
    return SimpleProgressBar(total or (len(iterable) if iterable is not None else 0), desc=desc)


# -------------------------- Text Processing --------------------------


def preprocess_text(text: str) -> str:
    text = str(text)
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    is_english = ascii_chars > len(text) * 0.8 if text else False
    if is_english:
        text = text.lower()
        text = re.sub(r'[^\w\s]|[\d]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
    else:
        words = jieba.cut(text)
        text = ' '.join(words)
    return text


def _process_single_text(text: str) -> str:
    return preprocess_text(text)


def preprocess_texts_parallel(texts, n_jobs: int):
    start = time.time()
    total = len(texts)
    pbar = get_progress_bar(total=total, desc="文本预处理")
    try:
        if n_jobs is None:
            n_jobs = max(1, multiprocessing.cpu_count() - 1)
        if n_jobs > 1:
            with multiprocessing.Pool(processes=n_jobs) as pool:
                preprocessed = []
                for out in pool.imap(_process_single_text, texts, chunksize=100):
                    preprocessed.append(out)
                    pbar.update(1)
        else:
            preprocessed = []
            for t in texts:
                preprocessed.append(preprocess_text(t))
                pbar.update(1)
    finally:
        pbar.close()
    elapsed = time.time() - start
    return preprocessed, elapsed


# -------------------------- Vectorization --------------------------

def vectorize_texts(texts, max_features=100):
    start = time.time()
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(texts)
    elapsed = time.time() - start
    return X, vectorizer, elapsed


# -------------------------- Clustering --------------------------

def cluster_texts_cpu(X, n_clusters: int, random_state: int):
    start = time.time()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = kmeans.fit_predict(X)
    elapsed = time.time() - start
    return labels, kmeans, elapsed

def cluster_texts(X, n_clusters: int, random_state: int):
    clustering_time = 0.0
    labels, kmeans, elapsed = cluster_texts_cpu(X, n_clusters, random_state)
    clustering_time += elapsed
    return labels, kmeans, clustering_time


# -------------------------- Silhouette Coefficient --------------------------

def calculate_silhouette_all(X, labels, batch_size=100):
    start = time.time()
    unique = np.unique(labels)
    if len(unique) < 2:
        return np.zeros(X.shape[0], dtype=float), time.time() - start
    for lb in unique:
        if np.sum(labels == lb) < 2:
            return np.zeros(X.shape[0], dtype=float), time.time() - start

    n_samples = X.shape[0]
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    label_to_idx = {lb: i for i, lb in enumerate(unique_labels)}
    cluster_sizes = np.array([(labels == lb).sum() for lb in unique_labels], dtype=int)

    s = np.zeros(n_samples, dtype=float)
    pbar = get_progress_bar(total=n_samples, desc="Computing silhouette coefficient")

    try:
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = np.arange(start_idx, end_idx)
            X_batch = X[batch_indices]

            D = pairwise_distances(X_batch, X, metric='cosine')

            means_to_clusters = np.zeros((len(batch_indices), len(unique_labels)), dtype=float)
            for j, lb in enumerate(unique_labels):
                mask = (labels == lb)
                if cluster_sizes[j] > 0:
                    means_to_clusters[:, j] = D[:, mask].mean(axis=1)

            self_cluster_col = np.array([label_to_idx[labels[i]] for i in batch_indices], dtype=int)
            size_self = cluster_sizes[self_cluster_col]
            sum_self = means_to_clusters[np.arange(len(batch_indices)), self_cluster_col] * size_self
            a = sum_self / np.maximum(size_self - 1, 1)

            means_masked = means_to_clusters.copy()
            means_masked[np.arange(len(batch_indices)), self_cluster_col] = np.inf
            b = means_masked.min(axis=1)

            denom = np.maximum(a, b)
            s_batch = np.where(denom > 0, (b - a) / denom, 0.0)

            s[batch_indices] = s_batch
            pbar.update(len(batch_indices))
    finally:
        pbar.close()

    elapsed = time.time() - start
    return s, elapsed


def normalize_scores(arr_or_val):
    return (np.asarray(arr_or_val, dtype=float) + 1.0) / 2.0


# -------------------------- Pipeline Functions --------------------------

def run_silhouette_pipeline(texts, n_clusters: int, n_jobs: int, batch_size: int):
    # Step 1: Preprocess texts
    preprocessed, t_pre = preprocess_texts_parallel(texts, n_jobs)
    # Step 2: Vectorize texts    
    X, vectorizer, t_vec = vectorize_texts(preprocessed)
    # Step 3: Cluster texts
    labels, kmeans, t_cluster = cluster_texts(X, n_clusters, random_state=42)

    # Step 3.1: Cluster result check and report
    if labels is None or len(labels) == 0:
        print("Warning: Clustering failed, cannot calculate silhouette coefficient")
        vals = np.zeros(X.shape[0], dtype=float)
        t_sil = 0.0
    else:
        unique_labels = np.unique(labels)
        print(f"Clustering result: {len(unique_labels)} clusters")
        for lb in unique_labels:
            cnt = int(np.sum(labels == lb))
            print(f"  - Cluster {lb}: {cnt} samples")

        # If the number of clusters is less than 2, it is impossible to calculate the silhouette coefficient
        if len(unique_labels) < 2:
            print(f"Error: Only {len(unique_labels)} clusters, cannot calculate silhouette coefficient")
            vals = np.zeros(X.shape[0], dtype=float)
            t_sil = 0.0
        else:
            # Check if there are clusters with less than 2 samples
            has_small_cluster = False
            for lb in unique_labels:
                cnt = int(np.sum(labels == lb))
                if cnt < 2:
                    print(f"Warning: Cluster {lb} has only {cnt} samples, cannot calculate silhouette coefficient")
                    has_small_cluster = True
            if has_small_cluster:
                vals = np.zeros(X.shape[0], dtype=float)
                t_sil = 0.0
            else:
                # Calculate silhouette coefficient with progress
                vals, t_sil = calculate_silhouette_all(X, labels, batch_size=batch_size)

    perf = {
        "Preprocessing time": t_pre,
        "Vectorization time": t_vec,
        "Clustering time": t_cluster,
        "Silhouette calculation time": t_sil,
        "Total time": t_pre + t_vec + t_cluster + t_sil,
        "Number of parallel jobs": n_jobs if n_jobs is not None else max(1, multiprocessing.cpu_count() - 1),
    }

    return {
        "vectors": X,
        "labels": labels,
        "silhouette": vals,
        "silhouette_norm": normalize_scores(vals),
        "perf": perf,
        "n_clusters": n_clusters,
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Silhouette Coefficient Analysis Tool")
    parser.add_argument("input_file", help="Input Excel file path")
    parser.add_argument("-c", "--column", default="text", help="Text column name (default: text)")
    parser.add_argument("-k", "--clusters", type=int, default=3, help="Number of clusters (default: 3)")
    parser.add_argument("-j", "--jobs", type=int, default=-1, help="Number of parallel jobs (default: -1, use all cores)")
    parser.add_argument("-b", "--batch-size", type=int, default=100, help="Batch size for silhouette calculation (default: 100)")
    parser.add_argument("-o", "--output", help="Output file path for results")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate input file
    if not os.path.exists(args.input_file):
        print(f"Error: Input file does not exist: {args.input_file}")
        return

    print(f"Reading file: {args.input_file}")
    df = pd.read_excel(args.input_file)
    if args.column not in df.columns:
        print(f"Error: Column '{args.column}' does not exist. Available columns: {', '.join(df.columns)}")
        return

    texts = df[args.column].astype(str).tolist()
    print(f"Number of samples: {len(texts)}")

    # Run pipeline
    res = run_silhouette_pipeline(
        texts=texts,
        n_clusters=args.clusters,
        n_jobs=args.jobs,
        batch_size=args.batch_size,
    )

    # Write results
    result_df = df.copy()
    result_df["silhouette_coefficient"] = res["silhouette"]
    result_df["normalized_silhouette_coefficient"] = res["silhouette_norm"]
    result_df["number_of_clusters"] = res["n_clusters"]

    # Output statistics
    avg_s = float(result_df["silhouette_coefficient"].mean())
    avg_sn = float(result_df["normalized_silhouette_coefficient"].mean())
    print(f"Average silhouette coefficient: {avg_s:.4f}")
    print(f"Average normalized silhouette coefficient: {avg_sn:.4f}")

    if args.output:
        result_df.to_excel(args.output, index=False)
        print(f"Results saved to: {args.output}")

    # Performance statistics
    perf = res["perf"]
    print("\nPerformance statistics:")
    for k, v in perf.items():
        if isinstance(v, float):
            print(f"  - {k}: {v:.2f} seconds")
        else:
            print(f"  - {k}: {v}")

    print(f"Analysis completed, processed {len(result_df)} samples")

if __name__ == "__main__":
    exit(main())
