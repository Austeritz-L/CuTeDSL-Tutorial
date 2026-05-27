import cutlass
import cutlass.cute as cute

@cute.kernel
def kernel():
    # Get the x component of the thread index (y and z components are unused)
    tidx, _, _ = cute.arch.thread_idx()
    # Only the first thread (thread 0) prints the message
    if tidx == 0:
        cute.printf("Hello world")

@cute.jit
def hello_world():

    # Print hello world from host code
    cute.printf("hello world")

    # Launch kernel
    kernel().launch(
        grid=(1, 1, 1),   # Single thread block
        block=(32, 1, 1)  # One warp (32 threads) per thread block
    )

cutlass.cuda.initialize_cuda_context()
hello_world()