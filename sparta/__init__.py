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

# "The canonical way to share information across modules within a single program is to create a special module (often called config or cfg)." https://docs.python.org/3/faq/programming.html#how-do-i-share-global-variables-across-modules
# Reference on how python caches modules so that they are only loaded once: https://docs.python.org/3/library/sys.html#sys.modules
global default_pycuda_stream, default_stream
default_pycuda_stream = pycuda.driver.Stream()
default_stream = torch.cuda.Stream(stream_ptr=default_pycuda_stream.handle)


# From https://github.com/inducer/pycuda/blob/5cb2e1a32f330c2984e2c1bf0579022494381ef1/pycuda/autoinit.py
def _finish_up():
    pass


import atexit

atexit.register(_finish_up)
