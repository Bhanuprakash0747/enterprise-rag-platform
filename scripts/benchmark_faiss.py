import time

import faiss
import numpy as np

from app.rag.vector_store import InMemoryVectorStore


VECTOR_STORE_PATH = "data/index/vector_store.json"


# --------------------------------------------------
# Load our real embeddings
# --------------------------------------------------

vector_store = InMemoryVectorStore()

vector_store.load(
    VECTOR_STORE_PATH
)

records = vector_store.get_all()

base_vectors = np.array(
    [record.embedding for record in records],
    dtype="float32"
)

dimension = base_vectors.shape[1]

print("Base vectors:", base_vectors.shape)


# --------------------------------------------------
# Create synthetic benchmark vectors
# --------------------------------------------------

rng = np.random.default_rng(42)

copies_per_vector = 1000

synthetic_vectors = np.repeat(
    base_vectors,
    copies_per_vector,
    axis=0
)

noise = rng.normal(
    loc=0.0,
    scale=0.01,
    size=synthetic_vectors.shape
).astype("float32")

synthetic_vectors = (
    synthetic_vectors + noise
).astype("float32")


print(
    "Synthetic vectors:",
    synthetic_vectors.shape
)


# --------------------------------------------------
# Normalize vectors
# --------------------------------------------------

faiss.normalize_L2(
    synthetic_vectors
)


# --------------------------------------------------
# Exact index
# --------------------------------------------------

exact_index = faiss.IndexFlatIP(
    dimension
)

exact_index.add(
    synthetic_vectors
)

print(
    "Exact index vectors:",
    exact_index.ntotal
)


# --------------------------------------------------
# IVF index
# --------------------------------------------------

nlist = 7

quantizer = faiss.IndexFlatIP(
    dimension
)

ivf_index = faiss.IndexIVFFlat(
    quantizer,
    dimension,
    nlist,
    faiss.METRIC_INNER_PRODUCT
)

# IVF needs to learn its clusters.
ivf_index.train(
    synthetic_vectors
)

ivf_index.add(
    synthetic_vectors
)

print(
    "IVF index vectors:",
    ivf_index.ntotal
)


# --------------------------------------------------
# Multiple query evaluation
# --------------------------------------------------

top_k = 5

queries = base_vectors.copy()

print("\nMulti-query IVF Evaluation")


for nprobe in range(1, nlist + 1):

    ivf_index.nprobe = nprobe

    recalls = []
    latencies = []

    for query in queries:

        query = query.reshape(
            1,
            -1
        ).copy()

        faiss.normalize_L2(
            query
        )

        # ------------------------------
        # Exact ground truth
        # ------------------------------

        exact_scores, exact_ids = (
            exact_index.search(
                query,
                top_k
            )
        )

        ground_truth = set(
            exact_ids[0]
        )

        # ------------------------------
        # IVF search
        # ------------------------------

        start = time.perf_counter()

        ivf_scores, ivf_ids = (
            ivf_index.search(
                query,
                top_k
            )
        )

        elapsed = (
            time.perf_counter() - start
        )

        retrieved = set(
            ivf_ids[0]
        )

        intersection = (
            ground_truth.intersection(
                retrieved
            )
        )

        recall = (
            len(intersection) / top_k
        )

        recalls.append(
            recall
        )

        latencies.append(
            elapsed * 1000
        )

    average_recall = (
        np.mean(recalls)
    )

    average_latency = (
        np.mean(latencies)
    )

    print(
        f"\nnprobe={nprobe}"
    )

    print(
        f"Average Recall@{top_k}: "
        f"{average_recall:.2%}"
    )

    print(
        f"Average latency: "
        f"{average_latency:.4f} ms"
    )