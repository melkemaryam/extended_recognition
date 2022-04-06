# How to run the code

### 1. Start Virtual Environment (if necessary)

```
$ source ~/.bashrc
$ workon traffic_signs
```

### 2. Change directories

```
$ cd extended_recognition/code
```

### 3. Specify arguments:
There are 5 arguments that need to be added when runing the code. The `class Args()` can be seen in `arguments.py`. 

* `-m` or `--model` to add the path to the output model
* `-op` or `--optimiser` to add the optimisation methods to use (optional)
* `-d`or `--dataset` to add the path to the input dataset
* `-i`or `--images` to add the path to the testing directory containing images
* `-pr`or `--predictions` to add the path to the output predictions directory

### 4. Run line in shell

The framework can be run through a command in the terminal. The `main.py` file will be run with the correct arguments. 

The following code will train and test the neural network with the assigned dataset.

```
$ python3 main.py --model ../output/neural_net.model --dataset ../gtsrb --images ../gtsrb/Test --predictions ../predictions
```

When optimising the neural network, add the `optimiser` argument as shown below.

### 5. Run optimisations

#### Bayesian Optimisation:

```
$ python3 main.py --model ../output/bayesian.model --optimiser bayesian --dataset ../gtsrb --images ../gtsrb/Test --predictions ../predictions
```

#### Hyperband Optimisation:

```
$ python3 main.py --model ../output/hyperband.model --optimiser hyperband --dataset ../gtsrb --images ../gtsrb/Test --predictions ../predictions
```

#### Random Search:

```
$ python3 main.py --model ../output/random_search.model --optimiser random --dataset ../gtsrb --images ../gtsrb/Test --predictions ../predictions
```

Make sure to always provide the correct model path in combination with the correct optimiser. 


## Results

Let the code run through all epochs of the training process. The output in the shell will indicate the current state of the programme.

A classification report will appear with the prediction accuracy.

The precdiction will be continued and 30 images will be used to predict the correct label.

In the directory `extended_recognition/predictions` the images can be viewed with their predicted labels.

## Using the final model

Once the best accuracy could be achieved, the corresponding model is already saved in the `output` folder. In order to use the model to predict images only (hence, without training, testing, or optimising), the `predict.py` file can be run with the same arguments.

An example:

```
$ python3 predict.py --model ../output/neural_net.model --dataset ../gtsrb --images ../gtsrb/Test --predictions ../predictions
```

The predicted images can then be viewed in the assigned folder.


