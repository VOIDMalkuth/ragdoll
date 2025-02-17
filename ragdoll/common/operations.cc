#include "common/operations.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <iostream>

#include "gccl.h"
#include "glog/logging.h"
#include "mpi.h"

#include "common/global_state.h"
#include "common/param.h"

namespace ragdoll {

GlobalState global_state;

// Always use MPI_COMM_WORLD for communication
void RagdollMPIInit(int device_id) {
  LOG(INFO) << "Initializing ragdoll using MPI";
  MPI_Init(nullptr, nullptr);
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  global_state.rank = rank;
  global_state.n_peers = world_size;

  gccl::gcclUniqueId id;
  if (rank == 0) id = gccl::GetUniqueId();
  MPI_Bcast((void *)&id, sizeof(gccl::gcclUniqueId), MPI_BYTE, 0,
            MPI_COMM_WORLD);
  gccl::CommInitRank(&global_state.comm, world_size, id, rank, device_id);

  global_state.device_id = gccl::GetDeviceId(global_state.comm);
  global_state.initialized = true;
  LOG(INFO) << "Finished initialization";
}
void RagdollEnvInit(int device_id) {
  LOG(INFO) << "Initializing ragdoll using env variables";
  int rank = GetEnvParam<int>("RANK");
  int world_size = GetEnvParam<int>("WORLD_SIZE");
  std::string master = GetEnvParam<std::string>("MASTER_ADDR");
  int port = GetEnvParam<int>("PORT");
  global_state.rank = rank;
  global_state.n_peers = world_size;
  gccl::gcclUniqueId id = gccl::GetUniqueId(master.c_str(), port, rank == 0);
  gccl::CommInitRank(&global_state.comm, world_size, id, rank, device_id);
  global_state.device_id = gccl::GetDeviceId(global_state.comm);
  global_state.initialized = true;
  LOG(INFO) << "Finished initialization";
}

void RagdollInit(int device_id) {
  if (global_state.initialized) return;
  if (GetEnvParam("USE_MPI", 1) == 1) {
    RagdollMPIInit(device_id);
  } else {
    RagdollEnvInit(device_id);
  }
}

void RagdollDeinit() {
  if (!global_state.initialized) return;

  gccl::FreeCommInfo(global_state.info);
  global_state.info = nullptr;

  gccl::CommDestroy(global_state.comm);
  global_state.comm = nullptr;

  if (GetEnvParam("USE_MPI", 1) == 1) {
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
  }

  global_state.initialized = false;
}

void RagdollPartitionGraph(int n_nodes, int *xadj, int *adjncy, int *sgn,
                           int **sg_xadj, int **sg_adjncy) {
  if (global_state.info != nullptr) {
    gccl::FreeCommInfo(global_state.info);
  }
  gccl::PartitionGraph(global_state.comm, n_nodes, xadj, adjncy,
                       &global_state.info, sgn, sg_xadj, sg_adjncy);
}

void RagdollPartitionGraphOnDir(const char *dirname, int *sgn, int **sg_xadj,
                                int **sg_adjncy) {
  if (global_state.info != nullptr) {
    gccl::FreeCommInfo(global_state.info);
  }
  gccl::PartitionGraph(global_state.comm, dirname, &global_state.info, sgn,
                       sg_xadj, sg_adjncy);
}

void RagdollLocalGraphDetailedInfo(int *global_nodes, int *local_owned_nodes, int *remote_owned_nodes,
                                   int **graph_parts, int **my_local_to_remote_mapping) {
  gccl::LocalGraphDetailedInfo(global_state.comm, global_nodes, local_owned_nodes, remote_owned_nodes, graph_parts, my_local_to_remote_mapping);
  CHECK(global_state.initialized) << "Cannot get LocalGraphDetailedInfo because ragdoll is not initialized";
}

int RagdollRank() {
  CHECK(global_state.initialized)
      << "Cannot get rank because ragdoll is not initialized";
  return global_state.rank;
}

void RagdollInitLogs(const char *file) { gccl::InitLogs(file); }
void RagdollDeInitLogs() { gccl::DeInitLogs(); }

int RagdollDeviceId() {
  CHECK(global_state.initialized)
      << "Cannot get rank because ragdoll is not initialized";
  return global_state.device_id;
}

int RagdollWorldSize() {
  CHECK(global_state.initialized)
      << "Cannot get rank because ragdoll is not initialized";
  return global_state.n_peers;
}

void RagdollRelease(int *ptr) { delete ptr; }

void RagdollDispatchFloat(float *data, int feat_size, int local_n_nodes,
                          float *local_data, int no_remote) {
  gccl::DispatchFloat(global_state.comm, data, feat_size, local_n_nodes,
                      local_data, no_remote);
}

void RagdollDispatchInt(int *data, int feat_size, int local_n_nodes,
                        int *local_data, int no_remote) {
  gccl::DispatchInt(global_state.comm, data, feat_size, local_n_nodes,
                    local_data, no_remote);
}

int RagdollGetLocalNNodes() { return gccl::GetLocalNNodes(global_state.comm); }

extern "C" {

void ragdoll_hello() { printf("Hello\n"); }

void ragdoll_init(int device_id) { RagdollInit(device_id); }
void ragdoll_deinit() { RagdollDeinit(); }
void ragdoll_init_logs(const char *file) { RagdollInitLogs(file); }
void ragdoll_deinit_logs() { RagdollDeInitLogs(); }
int ragdoll_rank() { return RagdollRank(); }
int ragdoll_device_id() { return RagdollDeviceId(); }
int ragdoll_world_size() { return RagdollWorldSize(); }

void ragdoll_partition_graph(int n_nodes, int *xadj, int *adjncy, int *sgn,
                             int **sg_xadj, int **sg_adjncy) {
  RagdollPartitionGraph(n_nodes, xadj, adjncy, sgn, sg_xadj, sg_adjncy);
}
void ragdoll_partition_graph_on_dir(const char *dirname, int *sgn,
                                    int **sg_xadj, int **sg_adjncy) {
  RagdollPartitionGraphOnDir(dirname, sgn, sg_xadj, sg_adjncy);
}

void ragdoll_local_graph_detailed_info(int *global_nodes, int *local_owned_nodes, int *remote_owned_nodes,
                                   int **graph_parts, int **my_local_to_remote_mapping) {
    RagdollLocalGraphDetailedInfo(global_nodes, local_owned_nodes, remote_owned_nodes,
                                  graph_parts, my_local_to_remote_mapping);
}

void ragdoll_release(int *ptr) { RagdollRelease(ptr); }
void ragdoll_dispatch_float(float *data, int feat_size, int local_n_nodes,
                            float *local_data, int no_remote) {
  RagdollDispatchFloat(data, feat_size, local_n_nodes, local_data, no_remote);
}

void ragdoll_dispatch_int(int *data, int feat_size, int local_n_nodes,
                          int *local_data, int no_remote) {
  RagdollDispatchInt(data, feat_size, local_n_nodes, local_data, no_remote);
}

int ragdoll_get_local_n_nodes() { return RagdollGetLocalNNodes(); };
}

void ExecuteGraphAllgather(std::shared_ptr<Tensor> tensor, int feat_size,
                           cudaStream_t stream) {
  // TODO Customize cuda stream
  gccl::GraphAllgather(global_state.comm, global_state.info, tensor->GetData(),
                       tensor->GetGCCLDataType(), feat_size, stream);
}

void ExecuteGraphAllgatherBackward(std::shared_ptr<Tensor> tensor,
                                   int feat_size, cudaStream_t stream) {
  // TODO Customize cuda stream
  gccl::GraphAllgatherBackward(global_state.comm, global_state.info,
                               tensor->GetData(), tensor->GetGCCLDataType(),
                               feat_size, stream);
}
}  // namespace ragdoll
