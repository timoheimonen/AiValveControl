# AiValveControl
An intelligent valve control system utilizing the PPO model for temperature management. This project simulates an environment and uses reinforcement learning to achieve optimal performance.

## License
This project is licensed under the MIT License. For more details, see the [LICENSE](LICENSE) file.

## Usage
`python ./train_ai.py` - Train the model. The model uses `train_ai_data.py`.

`python ./test_ai_gfx.py` - Graphical simulation of the model's functionality using Pygame. The model uses `train_ai_data.py`.

`python ./test_ai.py` - Console testing of the model.

Adjust the model training parameters in the PPO section of `train_ai.py`.
Adjust valve delay, temperatures, etc., in `train_ai_data.py`. These settings are used for both training and testing.

## Environment

Miniconda environment

Python version 3.8.20

Open AI Gym

Stable Baselines 3