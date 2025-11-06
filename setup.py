"""
Setup and Installation Script
Run this first to set up the anomaly detection module
"""

import os
import sys
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def create_directories():
    """Create necessary directories"""
    print_header("Creating Directory Structure")
    
    directories = [
        'weights',
        'logs',
        'data/videos/normal',
        'data/videos/anomalous',
    ]
    
    for dir_path in directories:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {dir_path}")
    
    print("\n✅ Directory structure created successfully!")


def check_dependencies():
    """Check if required packages are installed"""
    print_header("Checking Dependencies")
    
    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'tqdm': 'TQDM'
    }
    
    missing = []
    installed = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            installed.append(name)
            print(f"✓ {name} - Installed")
        except ImportError:
            missing.append(name)
            print(f"✗ {name} - NOT INSTALLED")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True


def check_cuda():
    """Check CUDA availability"""
    print_header("Checking CUDA/GPU Support")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"✓ CUDA is available!")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU")
            print("  Training will be slower on CPU")
            return False
    except ImportError:
        print("✗ PyTorch not installed - cannot check CUDA")
        return False


def test_imports():
    """Test if all modules can be imported"""
    print_header("Testing Module Imports")
    
    modules_to_test = [
        ('models', 'Model architectures'),
        ('data', 'Data processing'),
        ('utils', 'Utilities'),
        ('inference', 'Inference pipeline'),
        ('config', 'Configuration')
    ]
    
    all_success = True
    
    for module, description in modules_to_test:
        try:
            __import__(module)
            print(f"✓ {module:15s} - {description}")
        except Exception as e:
            print(f"✗ {module:15s} - ERROR: {e}")
            all_success = False
    
    if all_success:
        print("\n✅ All modules imported successfully!")
    else:
        print("\n⚠️  Some modules failed to import")
    
    return all_success


def show_next_steps():
    """Display next steps"""
    print_header("Next Steps")
    
    steps = """
1. Prepare Your Data
   ─────────────────
   Place your surveillance videos in:
     • data/videos/normal/     - Normal footage
     • data/videos/anomalous/  - Anomalous footage

2. Train the Autoencoder
   ───────────────────────
   python training/train_autoencoder.py --data_dir data/videos

3. Train the CNN-LSTM
   ──────────────────
   python training/train_cnn_lstm.py --data_dir data/videos

4. Run Inference
   ─────────────
   python demo.py

5. Integrate with Backend (Member 4)
   ─────────────────────────────────
   Update backend URL in config.py, then use:
   from inference.integration import BackendIntegration

For more details, see:
  • GUIDE.md - Complete usage guide
  • README.md - Project overview
  • demo.py - Working examples
    """
    
    print(steps)


def create_sample_config():
    """Create a sample configuration file if needed"""
    print_header("Configuration")
    
    config_path = Path('config.py')
    
    if config_path.exists():
        print("✓ config.py already exists")
        print("\nCurrent configuration:")
        print("  Frame Size: 224x224")
        print("  Sequence Length: 16 frames")
        print("  Anomaly Threshold: 0.7")
        print("\nEdit config.py to customize parameters")
    else:
        print("⚠️  config.py not found - this should not happen!")


def main():
    """Main setup function"""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           ANOMALY DETECTION MODULE - SETUP SCRIPT                ║
║                   Member 3 - CU Hackathon                        ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n" + "="*70)
        print("  ⚠️  SETUP INCOMPLETE - Install missing dependencies first")
        print("="*70)
        print("\nRun: pip install -r requirements.txt")
        return
    
    # Step 3: Check CUDA
    has_cuda = check_cuda()
    
    # Step 4: Test imports
    imports_ok = test_imports()
    
    # Step 5: Show configuration
    create_sample_config()
    
    # Final summary
    print("\n" + "="*70)
    if imports_ok and deps_ok:
        print("  ✅ SETUP COMPLETE - Ready to start!")
        print("="*70)
        
        if not has_cuda:
            print("\n  ⚠️  Note: No GPU detected - training will use CPU")
        
        show_next_steps()
    else:
        print("  ⚠️  SETUP INCOMPLETE - Please fix errors above")
        print("="*70)
    
    print("\n💡 Tip: Run 'python demo.py' to see a complete overview\n")


if __name__ == "__main__":
    main()
