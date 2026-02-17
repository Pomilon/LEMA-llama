import subprocess
import os
import argparse
import sys

def retrieve_kernel_logs(kernel_slug: str, output_dir: str = "./kaggle_logs"):
    """Retrieve kernel output and logs."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Retrieving logs for {kernel_slug} to {output_dir}...")
    
    # kaggle kernels output username/kernel-slug -p output_dir
    result = subprocess.run(
        ["kaggle", "kernels", "output", kernel_slug, "-p", output_dir],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✅ Logs retrieved successfully.")
        
        # Check for stderr
        error_log = os.path.join(output_dir, f"{kernel_slug.split('/')[-1]}.log")
        # Kaggle might output to notebook name.log or similar.
        # Usually it downloads the output files (output.log, etc.)
        
        files = os.listdir(output_dir)
        print(f"Files downloaded: {files}")
        
    else:
        print("❌ Failed to retrieve logs")
        print(result.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve Kaggle Kernel Logs")
    parser.add_argument("kernel_slug", help="Kernel slug (e.g., username/kernel-slug)")
    parser.add_argument("--output_dir", default="./kaggle_logs", help="Directory to save logs")
    
    args = parser.parse_args()
    retrieve_kernel_logs(args.kernel_slug, args.output_dir)
