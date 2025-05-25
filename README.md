# aa273-final-project
Repo for final project for Stanford AA273

# Install Necessary Packages

## Prerequisites
- [UV](https://docs.astral.sh/uv/getting-started/installation/) package manager
- Python 3.12+ (managed with pyenv or system Python)

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shiverrrai/aa273-final-project.git
   cd aa273-final-project
   ```

2. **Create and activate virtual environment:**
   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import numpy, plotly, scipy, matplotlib; print('All packages installed successfully!')"
   ```

## Deactivate Environment
When you're done working:
```bash
deactivate
```

## Reactivate Environment
To return to work on the project:
```bash
cd aa273-final-project
source .venv/bin/activate
```

