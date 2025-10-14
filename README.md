# Choose Your Own Adventure (CYOA) Llama Agents

This Python project runs local Llama models as agents to create an immersive choose your own adventure experience.

## Features
- One agent for overall response and collecting dialogue from character agents
- Another agent to decide what part of the response each character gets to read

## Setup
1. Ensure you have Python 3.8+
2. Install dependencies: `pip install -r requirements.txt`

## Usage
Run the main script:
```
python main.py
```

## Testing
Run tests with:
```
python -m unittest discover tests
```

## Notes
- Llama model files must be downloaded separately and placed in the `models/` directory.
- This is a basic scaffold. Replace placeholder logic with your own adventure and agent logic.
