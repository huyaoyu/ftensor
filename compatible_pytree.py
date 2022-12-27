
# At some version, PyTorch does not have the tree_map function.
# This is the case on Jetson device with Jetpack 4.6, where the 
# version of Python is 3.6 and PyTorch is 1.8.
# Try to make a compatible version instead. The code si copied from
# PyTorch 1.12.1.

try:
    from torch.utils._pytree import tree_map
except ImportError:
    from typing import Any
    from torch.utils._pytree import ( PyTree, tree_flatten, tree_unflatten)

    def tree_map(fn: Any, pytree: PyTree) -> PyTree:
        flat_args, spec = tree_flatten(pytree)
        return tree_unflatten([fn(i) for i in flat_args], spec)
