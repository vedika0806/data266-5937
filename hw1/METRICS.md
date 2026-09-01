
## HW1 Results — HP_ID=3 (Modified config: {'hidden_layers': [64, 32], 'lr': 0.0003, 'epochs': 30})

| Framework  | Model    | Mean Test Acc | Std Test Acc |
|------------|----------|----------------|----------------|
| PyTorch    | Baseline | 0.7427 | 0.0353 |
| PyTorch    | Modified | 0.6550 | 0.0149 |
| TensorFlow | Baseline | 0.7953 | 0.0041 |
| TensorFlow | Modified | 0.7719 | 0.0248 |


| Matrix size | CPU (ms)     | GPU kernel (ms) | H2D+D2H (ms) | Speedup   |
|-------------|--------------|------------------|--------------|-----------|
| 256         | 22.008       | 0.028            | 0.442        | 46.821    |
| 1024        | 3690.363     | 0.890            | 5.115        | 614.516   |
| 4096        | 693369.540   | 58.827           | 74.463       | 5201.962  |


**All detailed analysis for both the Neural Network (PyTorch/TensorFlow) and CUDA questions is provided in the respective Colab notebooks (`neural_networks.ipynb` and `cuda.ipynb`), including code, plots, and written interpretation. This file summarizes the key results and conclusions.**
