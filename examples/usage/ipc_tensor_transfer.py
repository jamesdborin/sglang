import torch
import torch.multiprocessing as mp
import os
import time

def run_worker(rank, queue):
    """
    Worker function to demonstrate CUDA IPC tensor transfer.
    
    Args:
        rank (int): The rank of the worker (0 or 1).
        queue (mp.Queue): The multiprocessing queue for communication.
    """
    try:
        # Set the device for this process
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        
        print(f"[Worker {rank}] Started on {device} (PID: {os.getpid()})")

        if rank == 0:
            # Worker 0: Create a tensor on GPU 0
            # We create a large enough tensor to make the transfer meaningful
            tensor_size = (1024, 1024)
            tensor = torch.ones(tensor_size, device=device) * 42.0
            print(f"[Worker {rank}] Created tensor on {device} with shape {tensor_size}")
            
            # Share the tensor via the queue
            # torch.multiprocessing.Queue handles CUDA IPC automatically:
            # 1. It gets the IPC handle of the CUDA tensor.
            # 2. It sends the handle through the queue.
            # 3. The tensor data remains on GPU 0.
            queue.put(tensor)
            print(f"[Worker {rank}] Sent tensor (IPC handle) to queue")
            
            # Wait for confirmation from Worker 1
            msg = queue.get()
            print(f"[Worker {rank}] Received message: {msg}")
            
        else:
            # Worker 1: Receive the tensor
            print(f"[Worker {rank}] Waiting for tensor from queue...")
            
            # This retrieves the tensor using the IPC handle sent by Worker 0
            # The received_tensor object is a wrapper around the memory on GPU 0
            received_tensor = queue.get()
            
            print(f"[Worker {rank}] Received tensor.")
            print(f"[Worker {rank}] Tensor device: {received_tensor.device} (Should be cuda:0)")
            
            # Verify the tensor is indeed on GPU 0
            assert received_tensor.device.index == 0
            
            # Copy the tensor to the local GPU (GPU 1)
            # This triggers the actual data transfer across the PCIe/NVLink bus
            start_time = time.time()
            local_tensor = received_tensor.to(device)
            torch.cuda.synchronize()
            end_time = time.time()
            
            print(f"[Worker {rank}] Copied tensor to {local_tensor.device}")
            print(f"[Worker {rank}] Transfer took {end_time - start_time:.6f} seconds")
            
            # Verify content
            expected = torch.ones((1024, 1024), device=device) * 42.0
            if torch.allclose(local_tensor, expected):
                print(f"[Worker {rank}] Verification SUCCESS! Tensors match.")
            else:
                print(f"[Worker {rank}] Verification FAILED! Tensors do not match.")
                
            # Signal worker 0 to exit
            queue.put("Done")
            
    except Exception as e:
        print(f"[Worker {rank}] Error: {e}")
        # If something goes wrong, try to signal the other process to stop waiting
        try:
            queue.put("Error")
        except:
            pass
        raise

def main():
    print("Initializing CUDA IPC Tensor Transfer Example...")
    
    # Ensure we use 'spawn' for CUDA multiprocessing. 
    # 'fork' is not supported for CUDA.
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    if torch.cuda.device_count() < 2:
        print("Error: This example requires at least 2 GPUs to demonstrate cross-device IPC.")
        print(f"Available GPUs: {torch.cuda.device_count()}")
        return

    # Create a multiprocessing queue
    # We use the context to ensure the queue is compatible with 'spawn'
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()

    # Launch two processes
    processes = []
    for rank in range(2):
        p = ctx.Process(target=run_worker, args=(rank, queue))
        p.start()
        processes.append(p)

    # Wait for processes to complete
    for p in processes:
        p.join()
        
    print("Example finished.")

if __name__ == "__main__":
    main()
