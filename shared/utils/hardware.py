import subprocess
import platform


def get_cpu_info_linux():
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True, check=True)
        print("---- CPU Info (Linux) ----")
        print(result.stdout)
    except FileNotFoundError:
        print("lscpu command not found.")
    except subprocess.CalledProcessError as e:
        print(f"Error running lscpu: {e}")


def get_gpu_info_nvidia():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print("---- GPU Info (NVIDIA) ----")
        print(result.stdout)
    except FileNotFoundError:
        print("nvidia-smi command not found. NVIDIA drivers might not be installed or not in PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error running nvidia-smi: {e}")


if __name__ == "__main__":
    os_type = platform.system()
    print(f"Operating System: {os_type}")

    if os_type == "Linux":
        get_cpu_info_linux()
        get_gpu_info_nvidia() # Also try rocm-smi if you have AMD
    elif os_type == "Darwin": # macOS
        print("---- CPU Info (macOS) ----")
        subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'])
        subprocess.run(['sysctl', '-n', 'hw.ncpu'])
        # For GPU, check System Information manually or use more specific tools if available
    elif os_type == "Windows":
        print("---- CPU Info (Windows) ----")
        subprocess.run(['wmic', 'cpu', 'get', 'Name,NumberOfCores,NumberOfLogicalProcessors'], shell=True)
        print("---- GPU Info (Windows - NVIDIA Example) ----")
        try:
            subprocess.run(['nvidia-smi'], shell=True) # May need to ensure nvidia-smi is in PATH
        except FileNotFoundError:
            print("nvidia-smi not found. For GPU info, check Task Manager or DxDiag.")
    else:
        print(f"Unsupported OS for this script: {os_type}")

    # For PyTorch to check CUDA availability and GPU details:
    try:
        import torch
        if torch.cuda.is_available():
            print("\n---- PyTorch CUDA Info ----")
            print(f"CUDA Available: {torch.cuda.is_available()}")
            print(f"CUDA Version (PyTorch compiled with): {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memory Allocated: {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
                print(f"    Memory Cached:    {torch.cuda.memory_reserved(i)/1024**2:.2f} MB") # formerly memory_cached
                props = torch.cuda.get_device_properties(i)
                print(f"    Total Memory:     {props.total_memory/1024**2:.2f} MB")
                print(f"    Compute Capability: {props.major}.{props.minor}")
        else:
            print("\nPyTorch: CUDA is not available.")
    except ImportError:
        print("\nPyTorch is not installed.")