#!/usr/bin/env python
"""
Environment verification script for taxi-demand-prediction project
Run this after setup to ensure all dependencies are correctly installed
"""

import sys
import os

def verify_setup():
    """Verify project setup"""
    print("=" * 60)
    print("🚕 Taxi Demand Prediction - Environment Verification")
    print("=" * 60)
    
    # Check Python version
    print(f"\n✅ Python Version: {sys.version.split()[0]}")
    
    # Check virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    print(f"✅ Virtual Environment Active: {in_venv}")
    
    # Check key packages
    packages = {
        'pyspark': 'PySpark',
        'torch': 'PyTorch',
        'pytorch_lightning': 'PyTorch Lightning',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-Learn',
        'matplotlib': 'Matplotlib',
        'pytest': 'pytest'
    }
    
    print("\n📦 Package Versions:")
    all_installed = True
    for pkg_import, pkg_name in packages.items():
        try:
            mod = __import__(pkg_import)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✅ {pkg_name:<25} {version}")
        except ImportError:
            print(f"  ❌ {pkg_name:<25} NOT INSTALLED")
            all_installed = False
    
    # Check folder structure
    print("\n📁 Folder Structure:")
    folders = [
        'src', 'src/data', 'src/features', 'src/models', 
        'src/training', 'src/evaluation', 'src/utils',
        'config', 'data', 'scripts', 'tests', 'notebooks', 'reports', 'logs'
    ]
    
    for folder in folders:
        exists = "✅" if os.path.isdir(folder) else "❌"
        print(f"  {exists} {folder}")
    
    # Check key files
    print("\n📄 Configuration Files:")
    files = ['requirements.txt', 'config/config.yaml', 'config/spark_config.yaml', 'README.md']
    for file in files:
        exists = "✅" if os.path.isfile(file) else "❌"
        print(f"  {exists} {file}")
    
    # Final status
    print("\n" + "=" * 60)
    if all_installed:
        print("✨ Setup complete! Ready for Week 1.2 (Data Ingestion)")
        print("=" * 60)
        return 0
    else:
        print("⚠️  Some packages are missing. Run: pip install -r requirements.txt")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(verify_setup())
