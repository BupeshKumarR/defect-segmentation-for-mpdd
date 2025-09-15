#!/bin/bash
# Activation script for MPDD project environment

echo "🔬 Activating MPDD Project Environment"
echo "======================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python -m venv venv"
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

echo "✅ Virtual environment activated"
echo "📍 Python location: $(which python)"
echo "📍 Pip location: $(which pip)"
echo ""
echo "🚀 Ready to work on MPDD project!"
echo "Run 'python demo.py' to see project overview"
echo "Run 'deactivate' to exit the environment"
