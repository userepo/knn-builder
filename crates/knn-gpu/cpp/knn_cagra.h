/*
 * The C interface between knn-gpu and its C++ shim over cuVS's CAGRA build
 * (knn_cagra.cpp). src/gpu.rs declares the same items; both sides assert
 * the layout of knn_cagra_params.
 */
#ifndef KNN_CAGRA_H
#define KNN_CAGRA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* One GPU's raft resources, reused by every build on it. */
typedef struct knn_cagra_gpu knn_cagra_gpu;

/* Vector types. */
enum { KNN_CAGRA_F32 = 0, KNN_CAGRA_F16 = 1 };

/* Metrics. */
enum { KNN_CAGRA_L2 = 0, KNN_CAGRA_INNER_PRODUCT = 1 };

/* Statuses: success, or the step that failed. */
enum {
  KNN_CAGRA_OK        = 0,
  KNN_CAGRA_RESOURCES = 1, /* creating the raft resources */
  KNN_CAGRA_DATASET   = 2, /* copying the vectors to the device */
  KNN_CAGRA_BUILD     = 3, /* the CAGRA build */
  KNN_CAGRA_GRAPH     = 4, /* a graph of another shape than asked for */
  KNN_CAGRA_COPY      = 5, /* copying the graph to the host */
  KNN_CAGRA_ARGUMENTS = 6, /* a null pointer, an empty shape, an unknown type */
};

/* The values the upstream service sets; the shim sets Faiss's own
 * defaults for the rest, as Faiss does. */
typedef struct knn_cagra_params {
  int32_t metric;
  uint32_t graph_degree;
  uint32_t intermediate_graph_degree;
  uint32_t n_lists;
  uint32_t kmeans_n_iters;
  uint32_t pq_bits;
  uint32_t pq_dim;
  uint32_t n_probes;
  double kmeans_trainset_fraction;
  float refinement_rate;
  bool force_random_rotation;
  bool conservative_memory_allocation;
} knn_cagra_params;

/*
 * Each function that can fail returns a status and, on failure, writes a
 * NUL-terminated message, truncated to fit, into `message` (`message_len`
 * bytes, may be 0).
 */

/*
 * Creates the resources; raft takes the device current at the first build.
 * Their workspace, what cuVS plans its batches by, is a quarter of the memory
 * of the device current now, at most 2 GiB.
 */
int knn_cagra_gpu_create(knn_cagra_gpu** gpu, char* message, size_t message_len);

/* Destroys resources from knn_cagra_gpu_create; null is ignored. */
void knn_cagra_gpu_destroy(knn_cagra_gpu* gpu);

/*
 * Builds the CAGRA graph of `rows` x `dimension` host vectors, row-major,
 * of type `dtype`, and copies it into `graph`, `graph_len` ids: rows x
 * params->graph_degree, row-major. Frees everything it allocated on the
 * device before returning, on failure too.
 */
int knn_cagra_build(knn_cagra_gpu* gpu,
                    const void* vectors,
                    int dtype,
                    int64_t rows,
                    int64_t dimension,
                    const knn_cagra_params* params,
                    uint32_t* graph,
                    size_t graph_len,
                    char* message,
                    size_t message_len);

#ifdef __cplusplus
}
#endif

#endif /* KNN_CAGRA_H */
