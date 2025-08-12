#!/bin/bash

echo "🚀 LSTM-MC Repository Creation Helper"
echo "====================================="
echo ""
echo "This script will help you create the LSTM-MC repository on GitHub and push your code."
echo ""

# Check if we're in the right directory
if [ ! -f "cuda_neutron_model.py" ]; then
    echo "❌ Error: Please run this script from the LSTM-TrajGAN directory"
    exit 1
fi

echo "✅ Current directory looks correct"
echo ""

# Check git status
echo "📊 Git Status:"
git status --porcelain
echo ""

# Check remote
echo "🔗 Current remote:"
git remote -v
echo ""

echo "📋 Next Steps:"
echo "1. Go to: https://github.com/kavyawadhwa134"
echo "2. Click 'New' (green button)"
echo "3. Repository name: LSTM-MC"
echo "4. Description: CUDA-Accelerated LSTM Trajectory GAN for neutron trajectory generation"
echo "5. Make it Public or Private (your choice)"
echo "6. IMPORTANT: DO NOT check 'Add a README file', 'Add .gitignore', or 'Choose a license'"
echo "7. Click 'Create repository'"
echo ""
echo "After creating the repository, come back here and run:"
echo "   git push -u origin master"
echo ""
echo "Or run this script again to attempt the push:"
echo "   ./create_repo.sh push"
echo ""

# If argument is "push", try to push
if [ "$1" = "push" ]; then
    echo "🚀 Attempting to push to GitHub..."
    echo ""
    
    # Check if remote exists
    if git ls-remote origin >/dev/null 2>&1; then
        echo "✅ Repository exists! Pushing code..."
        git push -u origin master
    else
        echo "❌ Repository doesn't exist yet. Please create it on GitHub first."
        echo "   Go to: https://github.com/kavyawadhwa134"
        echo "   Create repository: LSTM-MC"
        echo "   Then run: ./create_repo.sh push"
    fi
fi

echo ""
echo "🎯 Your repository will be at: https://github.com/kavyawadhwa134/LSTM-MC"
echo "📚 Ready for GPU cluster use with CUDA acceleration!"
