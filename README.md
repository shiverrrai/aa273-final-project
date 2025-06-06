# aa273-final-project
Repo for final project for Stanford AA273

# Install Necessary Packages

## Prerequisites
- Python 3.9+ (managed with pyenv or system Python)

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shiverrrai/aa273-final-project.git
   cd aa273-final-project
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv <venv name>
   source venv/<venv name>/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
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
source .venv/<venv name>/bin/activate
```

## Running the simulation
To run the top-level simulation:
```bash
cd aa273-final-project/src
python3 simulation_runner.py -h # for help options
python3 simulation_runner.py --measurement-factor 2 --no-save --monte-carlo-bounces # example usage
```

