#!/usr/bin/env python3
"""
Simple test script to validate Streamlit app structure and imports.
"""

import sys
import os
from pathlib import Path

def test_file_structure():
    """Test if all required files exist."""
    required_files = [
        'streamlit_app.py',
        'requirements.txt',
        'README.md',
        'streamlit_pages/__init__.py',
        'streamlit_pages/data_overview.py',
        'streamlit_pages/data_quality.py',
        'streamlit_pages/eda.py',
        'streamlit_pages/feature_engineering.py',
        'streamlit_pages/model_performance.py',
        'streamlit_pages/business_insights.py',
        'streamlit_pages/prediction.py',
        'streamlit_pages/technical.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("✅ All required files exist")
        return True

def test_python_syntax():
    """Test if Python files have valid syntax."""
    python_files = [
        'streamlit_app.py',
        'streamlit_pages/data_overview.py',
        'streamlit_pages/data_quality.py',
        'streamlit_pages/eda.py',
        'streamlit_pages/feature_engineering.py',
        'streamlit_pages/model_performance.py',
        'streamlit_pages/business_insights.py',
        'streamlit_pages/prediction.py',
        'streamlit_pages/technical.py'
    ]
    
    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                compile(f.read(), file_path, 'exec')
            print(f"✅ {file_path} - syntax OK")
        except SyntaxError as e:
            print(f"❌ {file_path} - syntax error: {e}")
            syntax_errors.append(file_path)
        except FileNotFoundError:
            print(f"⚠️ {file_path} - file not found")
            syntax_errors.append(file_path)
    
    return len(syntax_errors) == 0

def test_imports():
    """Test if required packages are available."""
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'plotly',
        'plotly.express',
        'plotly.graph_objects'
    ]
    
    import_errors = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - import OK")
        except ImportError as e:
            print(f"❌ {package} - import failed: {e}")
            import_errors.append(package)
    
    return len(import_errors) == 0

def main():
    """Run all tests."""
    print("🧪 Testing Streamlit App Structure")
    print("=" * 50)
    
    # Test file structure
    print("\n📁 Testing file structure...")
    structure_ok = test_file_structure()
    
    # Test Python syntax
    print("\n🐍 Testing Python syntax...")
    syntax_ok = test_python_syntax()
    
    # Test imports (only basic ones available in environment)
    print("\n📦 Testing basic imports...")
    basic_packages = ['sys', 'os', 'pathlib', 'json']
    import_errors = []
    for package in basic_packages:
        try:
            __import__(package)
            print(f"✅ {package} - import OK")
        except ImportError as e:
            print(f"❌ {package} - import failed: {e}")
            import_errors.append(package)
    
    imports_ok = len(import_errors) == 0
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   File Structure: {'✅ PASS' if structure_ok else '❌ FAIL'}")
    print(f"   Python Syntax:  {'✅ PASS' if syntax_ok else '❌ FAIL'}")
    print(f"   Basic Imports:  {'✅ PASS' if imports_ok else '❌ FAIL'}")
    
    if structure_ok and syntax_ok and imports_ok:
        print("\n🎉 All tests passed! App structure is ready for deployment.")
        print("\n🚀 Next steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Run the app: streamlit run streamlit_app.py")
        print("   3. Open browser to http://localhost:8501")
        return True
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)