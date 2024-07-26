# Color Style Transfer with Modulated Flows

<p align="center">
     <img src="./img/results_unsplash.png" style="width: 1000px"/>
</p>

This is the source code of the paper 
["Color Style Transfer with Modulated Flows"](https://openreview.net/forum?id=Lztt4WVusu) 
for Workshop SPIGM @ ICML 2024.

Please refer to the
- <strong>src</strong> directory for models definitions
- <strong>generate_flows_v2</strong> script for training the dataset of rectified flows
- <strong>train_encoder_v2</strong> script for training the encoder

Call `python3 run_inference.py --help` to see a full list of arguments for inference.
`Ctrl+C` cancels the execution.

<p align="center">
     <img src="./img/SPIGM_visual_abstract.png" style="width: 500px"/>
</p>