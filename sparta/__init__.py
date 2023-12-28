# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

try:
    import torch
    import pycuda
except ImportError:
    raise ImportError(
        "Our SparTA variant requires PyTorch and PyCuda to be installed."
    )
finally:
    __env_ready__ = torch is not None and pycuda is not None


from sparta import nn, testing
import pycuda.autoprimaryctx
import sys

# This is a pointer to the module object instance itself.
# Reference: https://stackoverflow.com/a/35904211/5555077
this = sys.modules[__name__]
this.current_pycuda_stream: dict[torch.device, pycuda.driver.Stream]
this.current_stream: dict[torch.device, torch.cuda.Stream]


def simple_initialize_current_streams():
    """
    The canonical way to refer to these current streams objects are
    import sparta
    sparta.current_stream
    Reference: https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
    Reference on how python caches modules so that they are only loaded once: https://docs.python.org/3/library/sys.html#sys.modules
    """
    print(
        "Warning: intrasm_engine not installed. Initializing current streams"
        " using the simple function provided by sparta, which may be obsolete."
    )
    num_gpus = torch.cuda.device_count()
    this.current_pycuda_stream = {}
    this.current_stream = {}
    for i in range(num_gpus):
        this.current_pycuda_stream[
            torch.device(f"cuda:{i}")
        ] = pycuda.driver.Stream()
        this.current_stream[torch.device(f"cuda:{i}")] = torch.cuda.Stream(
            stream_ptr=this.current_pycuda_stream[
                torch.device(f"cuda:{i}")
            ].handle
        )

    torch.cuda.set_device(0)
    torch.cuda.set_stream(this.current_stream[torch.device(f"cuda:{0}")])


try:
    import intrasm_engine

    this.current_pycuda_stream = intrasm_engine.current_pycuda_stream
    this.current_stream = intrasm_engine.current_stream
except ImportError:
    simple_initialize_current_streams()


# From https://github.com/inducer/pycuda/blob/5cb2e1a32f330c2984e2c1bf0579022494381ef1/pycuda/autoinit.py
def _finish_up():
    pass


import atexit

atexit.register(_finish_up)
