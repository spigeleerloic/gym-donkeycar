# How to use consumption car efficient model ?


in the $src$ folder you will find multiple scripts :

- $play.py$ is a script for manual driving (helps debugging information)

You can also play by using the simulator in itself but you will not get the information at each state.

- $train.py$ is the script used for training purposes.

This script will generate logs and model checkpoints in order to get relevant information from the training. It will also redirect some logs to wandb that will create some figures of the evolution of rewards,...

- $test.py$ is the script that can use previously trained models to get an overview of the efficiency of the model after training. 