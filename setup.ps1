# Hand Pose Estimation - Setup Script
# Run this script to set up the environment and test the installation

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "3D Hand Pose Estimation - Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Check Python version
Write-Host "[1/4] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
    
    # Extract version number
    if ($pythonVersion -match "Python (\d+\.\d+)") {
        $version = [decimal]$matches[1]
        if ($version -lt 3.8) {
            Write-Host "  Warning: Python 3.8+ recommended (you have $version)" -ForegroundColor Red
        }
    }
} catch {
    Write-Host "  Error: Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Step 2: Create virtual environment
Write-Host ""
Write-Host "[2/4] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "  Virtual environment already exists, skipping..." -ForegroundColor Gray
} else {
    python -m venv venv
    if ($?) {
        Write-Host "  Virtual environment created successfully" -ForegroundColor Green
    } else {
        Write-Host "  Error creating virtual environment" -ForegroundColor Red
        exit 1
    }
}

# Step 3: Activate and install dependencies
Write-Host ""
Write-Host "[3/4] Installing dependencies..." -ForegroundColor Yellow
Write-Host "  This may take a few minutes..." -ForegroundColor Gray
Write-Host "  Note: Using MediaPipe 0.10.9 for Windows compatibility" -ForegroundColor Gray

# Activate virtual environment and install
& ".\venv\Scripts\Activate.ps1"
pip install --upgrade pip | Out-Null
pip install -r requirements.txt

if ($?) {
    Write-Host "  Dependencies installed successfully" -ForegroundColor Green
} else {
    Write-Host "  Error installing dependencies" -ForegroundColor Red
    exit 1
}

# Step 4: Run tests
Write-Host ""
Write-Host "[4/4] Running installation tests..." -ForegroundColor Yellow
python test_installation.py

# Final message
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Prepare a hand gesture image (JPG/PNG)" -ForegroundColor White
Write-Host "2. Run inference:" -ForegroundColor White
Write-Host "   python infer_hand_pose_v2.py your_image.jpg --visualize" -ForegroundColor Cyan
Write-Host ""
Write-Host "See QUICKSTART.md for usage examples" -ForegroundColor Gray
Write-Host ""
