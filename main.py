import sys
import subprocess

def check_dependencies():
    """Verify all tools are installed."""
    try:
        import ray
        ray.init(ignore_reinit_error=True)
        print(" Ray initialized")
    except ImportError:
        print("❌ Ray not found. Please run setup.sh again.")
        sys.exit(1)

    # Check Node/Circom via subprocess
    try:
        subprocess.run(["circom", "--version"], check=True, capture_output=True)
        print("Circom installed")
    except Exception:
        print("❌ Circom not found. Check setup.sh.")
        sys.exit(1)

def main():
    check_dependencies()
    print("🚀 Starting Federated Learning + ZKP Demo...")
    
    # Placeholder for your logic
    # In the real project, you would load your model and data here
    print("Environment is ready. Proceed with implementation.")

if __name__ == "__main__":
    main()