# neural-tube-imaging
Research with Yanrong Xiao and Alexander Smeulders through Scott Holley's lab at Yale University working on neural tube cell shape analysis in zebrafish.


### Installation

Requires python3.11 so if not already installed

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install ultrack
```

Always use uv in front of pip from now on.

Can also do via conda env