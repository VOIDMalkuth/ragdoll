import ctypes
import pathlib
import sysconfig
import time
import numpy as np
import torch

from ragdoll.perf_counter import perf_counter


def get_ext_suffix():
    """Determine library extension for various versions of Python."""
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix

    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix

    return '.so'


class RagdollCore(object):
    def __init__(self):
        lib_name = "ragdoll_torch_ops" + get_ext_suffix()
        so_path = pathlib.Path(__file__).with_name(lib_name)
        # so_path = str(pathlib.Path(__file__).with_name('libragdoll_core.so'))
        print('so path is ', so_path)
        self.lib = ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)

    def hello(self):
        self.lib.ragdoll_hello()

    def set_comm_pattern(self, comm):
        from . import torch
        torch.set_comm_pattern(comm)

    def init_logs(self, log_file):
        log_file = log_file.encode('utf-8')
        self.lib.ragdoll_init_logs(log_file)

    def init(self, device_id=-1):
        self.lib.ragdoll_init(device_id)

    def rank(self):
        return self.lib.ragdoll_rank()

    def device_id(self):
        return self.lib.ragdoll_device_id()

    def world_size(self):
        return self.lib.ragdoll_world_size()

    def partition_graph(self, n_nodes, xadj, adjncy):
        c_xadj = (ctypes.c_int * len(xadj))(*xadj)
        c_adjncy = (ctypes.c_int * len(adjncy))(*adjncy)
        sg_n = ctypes.c_int32()
        sg_xadj = ctypes.POINTER(ctypes.c_int)()
        sg_adjncy = ctypes.POINTER(ctypes.c_int)()
        self.lib.ragdoll_partition_graph(n_nodes, c_xadj, c_adjncy, ctypes.byref(
            sg_n), ctypes.byref(sg_xadj), ctypes.byref(sg_adjncy))

        n_edges = sg_xadj[sg_n.value]
        print('Subgraph nodes:', sg_n.value, 'local nodes',
              self.lib.ragdoll_get_local_n_nodes(), 'edges:', n_edges)
        py_sg_xadj = [sg_xadj[i] for i in range(sg_n.value + 1)]
        py_sg_adjncy = [sg_adjncy[i] for i in range(n_edges)]
        self.lib.ragdoll_release(sg_xadj)
        self.lib.ragdoll_release(sg_adjncy)
        return sg_n.value, py_sg_xadj, py_sg_adjncy

    def partition_graph_on_dir(self, dirname):
        sg_n = ctypes.c_int32()
        sg_xadj = ctypes.POINTER(ctypes.c_int)()
        sg_adjncy = ctypes.POINTER(ctypes.c_int)()
        dirname = dirname.encode('utf-8')
        self.lib.ragdoll_partition_graph_on_dir(dirname, ctypes.byref(
            sg_n), ctypes.byref(sg_xadj), ctypes.byref(sg_adjncy))

        n_edges = sg_xadj[sg_n.value]
        print('Subgraph nodes:', sg_n.value, 'local nodes',
              self.lib.ragdoll_get_local_n_nodes(), 'edges:', n_edges)
        py_sg_xadj = [sg_xadj[i] for i in range(sg_n.value + 1)]
        py_sg_adjncy = [sg_adjncy[i] for i in range(n_edges)]
        self.lib.ragdoll_release(sg_xadj)
        self.lib.ragdoll_release(sg_adjncy)
        return sg_n.value, py_sg_xadj, py_sg_adjncy

    def dispatch_float(self, t, feat_size, local_n_nodes, no_remote=0):
        perf_counter.record_start("dispatch_float_py")
        if t is None:
            t = np.array([])
        if isinstance(t, torch.Tensor):
            t = t.numpy()
        # t = t.ravel().tolist()
        # c_ptr = (ctypes.c_float * len(t))(*t)
        c_ptr = t.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        local_data = (ctypes.c_float * (local_n_nodes * feat_size))()

        perf_counter.record_start("dispatch_float_c")
        self.lib.ragdoll_dispatch_float(
            c_ptr, feat_size, local_n_nodes, local_data, no_remote)
        perf_counter.record_end("dispatch_float_c")
        
        dispatch_res = [local_data[v] for v in range(local_n_nodes * feat_size)]
        perf_counter.record_end("dispatch_float_py")
        return dispatch_res

    def dispatch_int(self, t, feat_size, local_n_nodes, no_remote=0):
        perf_counter.record_start("dispatch_int_py")

        if t is None:
            t = np.array([])
        if isinstance(t, torch.Tensor):
            t = t.numpy()
        # t = t.ravel().tolist()
        # c_ptr = (ctypes.c_int * len(t))(*t)
        c_ptr = t.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        local_data = (ctypes.c_int * (local_n_nodes * feat_size))()
        perf_counter.record_start("dispatch_int_c")
        self.lib.ragdoll_dispatch_float(
            c_ptr, feat_size, local_n_nodes, local_data, no_remote)
        perf_counter.record_end("dispatch_int_c")
        dispatch_res = [local_data[v] for v in range(local_n_nodes * feat_size)]

        perf_counter.record_end("dispatch_int_py")
        return dispatch_res

    def get_local_n_nodes(self):
        return self.lib.ragdoll_get_local_n_nodes()
