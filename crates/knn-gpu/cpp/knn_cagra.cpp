// The CAGRA build through cuVS's C++ API, the API Faiss builds through, so
// every IVF-PQ graph build and search setting reaches cuVS as the upstream
// service passes it; cuVS's C API fixes two of the search settings
// (docs/params.md).
//
// Device memory is owned by C++ objects that free it on every return, and
// every exception is caught and returned as a status with its message.

#include "knn_cagra.h"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/version_config.h>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <optional>

static_assert(CUVS_VERSION_MAJOR == 26 && CUVS_VERSION_MINOR == 8 && CUVS_VERSION_PATCH == 1,
              "docs/params.md checks every setting against libcuvs 26.08.01; "
              "recheck it before building against another version");

// The layout src/gpu.rs asserts too.
static_assert(offsetof(knn_cagra_params, metric) == 0);
static_assert(offsetof(knn_cagra_params, graph_degree) == 4);
static_assert(offsetof(knn_cagra_params, intermediate_graph_degree) == 8);
static_assert(offsetof(knn_cagra_params, n_lists) == 12);
static_assert(offsetof(knn_cagra_params, kmeans_n_iters) == 16);
static_assert(offsetof(knn_cagra_params, pq_bits) == 20);
static_assert(offsetof(knn_cagra_params, pq_dim) == 24);
static_assert(offsetof(knn_cagra_params, n_probes) == 28);
static_assert(offsetof(knn_cagra_params, kmeans_trainset_fraction) == 32);
static_assert(offsetof(knn_cagra_params, refinement_rate) == 40);
static_assert(offsetof(knn_cagra_params, force_random_rotation) == 44);
static_assert(offsetof(knn_cagra_params, conservative_memory_allocation) == 45);
static_assert(sizeof(knn_cagra_params) == 48);

// As cuVS's C API creates its resources handle (c/src/core/c_api.cpp).
struct knn_cagra_gpu {
  raft::resources res;
};

// The most workspace a build plans its batches by. RAFT's default is a
// quarter of the device's memory, and cuVS sizes batches from it (the IVF-PQ
// search's among them), so a larger GPU would batch, and take device memory,
// otherwise than the 8 GB GPU the service's memory model was measured on,
// whose quarter is about this.
constexpr std::size_t kWorkspaceLimit = std::size_t{2} << 30;

namespace {

void set_message(char* message, std::size_t len, const char* text)
{
  if (message != nullptr && len > 0) { std::snprintf(message, len, "%s", text); }
}

// Runs one step, returning `status` with the message of any exception.
template <typename Step>
int run(int status, char* message, std::size_t len, Step&& step)
{
  try {
    step();
    return KNN_CAGRA_OK;
  } catch (const std::exception& e) {
    set_message(message, len, e.what());
  } catch (...) {
    set_message(message, len, "unknown exception");
  }
  return status;
}

// The parameters as Faiss builds them: default-constructed, then each field
// Faiss sets (faiss/gpu/GpuIndexCagra.cu:63-92, faiss/gpu/impl/CuvsCagra.cu:
// 79-87, 187-195). The values the service sets come from `p`; Faiss's own
// defaults for the rest are set here, as Faiss sets them.
cuvs::neighbors::cagra::index_params faiss_index_params(const knn_cagra_params& p)
{
  namespace ivf_pq = cuvs::neighbors::ivf_pq;
  namespace cagra  = cuvs::neighbors::cagra;
  auto const metric = p.metric == KNN_CAGRA_INNER_PRODUCT
                        ? cuvs::distance::DistanceType::InnerProduct
                        : cuvs::distance::DistanceType::L2Expanded;

  ivf_pq::index_params build;
  build.metric                         = metric;
  build.n_lists                        = p.n_lists;
  build.kmeans_n_iters                 = p.kmeans_n_iters;
  build.kmeans_trainset_fraction       = p.kmeans_trainset_fraction;
  build.pq_bits                        = p.pq_bits;
  build.pq_dim                         = p.pq_dim;
  build.codebook_kind                  = ivf_pq::codebook_gen::PER_SUBSPACE;
  build.force_random_rotation          = p.force_random_rotation;
  build.conservative_memory_allocation = p.conservative_memory_allocation;

  ivf_pq::search_params search;
  search.n_probes                 = p.n_probes;
  search.lut_dtype                = CUDA_R_32F;
  search.preferred_shmem_carveout = 1.0;
  search.max_internal_batch_size  = 4096;
  // Faiss leaves these two at their C++ defaults; set to the same values so
  // every search setting is explicit. coarse_search_dtype and
  // max_internal_batch_size are the two cuVS's C API cannot pass.
  search.internal_distance_dtype = CUDA_R_32F;
  search.coarse_search_dtype     = CUDA_R_32F;

  cagra::graph_build_params::ivf_pq_params graph_build;
  graph_build.build_params    = build;
  graph_build.search_params   = search;
  graph_build.refinement_rate = p.refinement_rate;

  cagra::index_params params;
  params.metric                    = metric;
  params.intermediate_graph_degree = p.intermediate_graph_degree;
  params.graph_degree              = p.graph_degree;
  params.guarantee_connectivity    = false;
  params.graph_build_params        = graph_build;
  return params;
}

// Copies the host vectors to a padded device dataset, builds on it, and
// copies the graph back, as Faiss's CuvsCagra::train does for host data
// (faiss/gpu/impl/CuvsCagra.cu:214-223).
template <typename T>
int build_graph(raft::resources& res,
                const T* vectors,
                std::int64_t rows,
                std::int64_t dimension,
                const knn_cagra_params& p,
                std::uint32_t* graph,
                std::size_t graph_len,
                char* message,
                std::size_t len)
{
  // Declared before the index, which views it, so it is freed after.
  std::unique_ptr<cuvs::neighbors::device_padded_dataset<T, std::int64_t>> dataset;
  int status = run(KNN_CAGRA_DATASET, message, len, [&] {
    auto host = raft::make_host_matrix_view<const T, std::int64_t>(vectors, rows, dimension);
    dataset   = cuvs::neighbors::make_device_padded_dataset(res, host);
  });
  if (status != KNN_CAGRA_OK) { return status; }

  std::optional<cuvs::neighbors::cagra::device_padded_index<T, std::uint32_t>> index;
  status = run(KNN_CAGRA_BUILD, message, len, [&] {
    index.emplace(cuvs::neighbors::cagra::build(res, faiss_index_params(p), dataset->as_dataset_view()));
  });
  if (status != KNN_CAGRA_OK) { return status; }

  auto const view = index->graph();
  if (view.extent(0) != rows || view.extent(1) != static_cast<std::int64_t>(p.graph_degree) ||
      static_cast<std::size_t>(view.size()) != graph_len) {
    if (message != nullptr && len > 0) {
      std::snprintf(message,
                    len,
                    "%lld x %lld for %lld vectors of degree %u",
                    static_cast<long long>(view.extent(0)),
                    static_cast<long long>(view.extent(1)),
                    static_cast<long long>(rows),
                    p.graph_degree);
    }
    return KNN_CAGRA_GRAPH;
  }
  return run(KNN_CAGRA_COPY, message, len, [&] {
    raft::copy(graph, view.data_handle(), graph_len, raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
  });
}

}  // namespace

extern "C" int knn_cagra_gpu_create(knn_cagra_gpu** gpu, char* message, std::size_t message_len)
{
  if (gpu == nullptr) {
    set_message(message, message_len, "null output pointer");
    return KNN_CAGRA_ARGUMENTS;
  }
  *gpu = nullptr;
  return run(KNN_CAGRA_RESOURCES, message, message_len, [&] {
    auto created = std::make_unique<knn_cagra_gpu>();
    std::size_t free  = 0;
    std::size_t total = 0;
    RAFT_CUDA_TRY(cudaMemGetInfo(&free, &total));
    // RAFT's default resource, as its own factory takes it, with a limit.
    raft::resource::set_workspace_resource(
      created->res,
      raft::mr::device_resource{rmm::mr::get_current_device_resource_ref()},
      std::min(total / 4, kWorkspaceLimit));
    *gpu = created.release();
  });
}

extern "C" void knn_cagra_gpu_destroy(knn_cagra_gpu* gpu) { delete gpu; }

extern "C" int knn_cagra_build(knn_cagra_gpu* gpu,
                               const void* vectors,
                               int dtype,
                               std::int64_t rows,
                               std::int64_t dimension,
                               const knn_cagra_params* params,
                               std::uint32_t* graph,
                               std::size_t graph_len,
                               char* message,
                               std::size_t message_len)
{
  if (gpu == nullptr || vectors == nullptr || params == nullptr || graph == nullptr || rows <= 0 ||
      dimension <= 0 ||
      (params->metric != KNN_CAGRA_L2 && params->metric != KNN_CAGRA_INNER_PRODUCT)) {
    set_message(message, message_len, "null pointer, empty shape or unknown metric");
    return KNN_CAGRA_ARGUMENTS;
  }
  switch (dtype) {
    case KNN_CAGRA_F32:
      return build_graph(gpu->res,
                         static_cast<const float*>(vectors),
                         rows,
                         dimension,
                         *params,
                         graph,
                         graph_len,
                         message,
                         message_len);
    case KNN_CAGRA_F16:
      return build_graph(gpu->res,
                         static_cast<const half*>(vectors),
                         rows,
                         dimension,
                         *params,
                         graph,
                         graph_len,
                         message,
                         message_len);
    case KNN_CAGRA_I8:
      return build_graph(gpu->res,
                         static_cast<const std::int8_t*>(vectors),
                         rows,
                         dimension,
                         *params,
                         graph,
                         graph_len,
                         message,
                         message_len);
    default: set_message(message, message_len, "unknown vector type"); return KNN_CAGRA_ARGUMENTS;
  }
}
