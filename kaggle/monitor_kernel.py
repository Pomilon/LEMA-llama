import subprocess
import time
import json
import argparse
import sys
import os

def get_kernel_status(kernel_slug: str) -> str:
    """Get current kernel status."""
    result = subprocess.run(
        ["kaggle", "kernels", "status", kernel_slug],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        # Output format is usually: "kernel-slug has status \"status\""
        # Example: "kloyford/lema-finetuning-demo has status \"running\""
        output = result.stdout.strip()
        if "has status" in output:
            try:
                # Extract status from quotes
                status = output.split('has status')[1].strip().strip('"')
                return status
            except IndexError:
                pass
        return output # Return full output if parsing fails, might be just status
    
    return "unknown"

def monitor_kernel(kernel_slug: str, check_interval: int = 60):
    """Monitor kernel until completion or failure."""
    
    print(f"Monitoring kernel: {kernel_slug}")
    start_time = time.time()
    
    while True:
        status = get_kernel_status(kernel_slug)
        elapsed = time.time() - start_time
        print(f"[{elapsed:.0f}s] Status: {status}")
        
        if status == "complete":
            print("✅ Kernel completed successfully!")
            return True
        elif status in ["error", "failed", "cancelled"]:
            print(f"❌ Kernel failed with status: {status}")
            return False
        
        time.sleep(check_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor Kaggle Kernel")
    parser.add_argument("kernel_slug", help="Kernel slug (e.g., username/kernel-slug)")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    
    args = parser.parse_args()
    monitor_kernel(args.kernel_slug, args.interval)
