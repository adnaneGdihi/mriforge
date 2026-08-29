.. _module_loader:

Async Data Loading Infrastructure
=================================

**File Path:** ``src/infrastructure/data/async_data_loader.py``

High-Level Logic
----------------
This module implements a high-performance, asynchronous data loading pipeline designed to maximize GPU utilization. It decouples the CPU-bound data preparation (disk I/O, decompression, augmentation) from the GPU-bound model execution using a **Consumer-Producer** pattern.

Operational Mechanisms:
1.  **CUDA Streams:** Utilizes a dedicated `transfer_stream` to copy tensors to the GPU without blocking the default compute stream.
2.  **Pinned Memory:** Leverages page-locked (pinned) memory for faster Host-to-Device (H2D) DMA transfers.
3.  **Background Worker:** A daemon thread prefetches batches into a `queue`, ensuring the GPU never waits for Python overhead.

Algorithmic Flow
----------------

.. code-block:: python

   # Pseudocode of __next__
   if next_batch is ready:
       current = next_batch
       synchronize(transfer_stream) # Wait for transfer to complete
       trigger_prefetch_next()      # Start next transfer on side stream
       return current
   else:
       fetch_and_block()

Class Breakdown
---------------

.. autoclass:: mriforge.infrastructure.data.async_data_loader.AsyncDataLoader
   :members:
   :undoc-members:

.. autoclass:: mriforge.infrastructure.data.async_data_loader.CUDAStreamManager
   :members:
