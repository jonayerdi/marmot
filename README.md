# MarMot: Metamorphic Runtime Monitoring of Autonomous Driving Systems
This is the code used in the paper "MarMot: Metamorphic Runtime Monitoring of Autonomous
Driving Systems". For replicating our study, the dataset we use can be found at [https://doi.org/10.5281/zenodo.8202316](https://doi.org/10.5281/zenodo.8202316).

It contains the following artifacts:

* `evaluation_scripts` Contains the code used for evaluating monitoring approaches (as well as other stuff, like training SelfOracle models).
* `leorover_scripts` Contains the code ran in the LeoRover vehicle for running ADS models, injecting anomalies, and recording datasets. Most of this is modified from code provided by the LeoRover vendor. It also contains the scripts for training ADS models and generating the mutants.

In addition to our own code, these artifacts contain modified code from the following sources:

* SelfOracle
    * [https://github.com/testingautomated-usi/selforacle](https://github.com/testingautomated-usi/selforacle) (MIT License)
* LeoRover
    * [https://github.com/LeoRover/leo_examples/tree/master/leo_example_line_follower](https://github.com/LeoRover/leo_examples/tree/master/leo_example_line_follower) (MIT License)
* Anomalies
    * [https://github.com/tsigalko18/transferability-testing-sdcs/blob/main/visualodometry/corruptions/hendrycks.py](https://github.com/tsigalko18/transferability-testing-sdcs/blob/main/visualodometry/corruptions/hendrycks.py) (MIT License)
    * [https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_cifar_c.py](https://github.com/hendrycks/robustness/blob/master/ImageNet-C/create_c/make_cifar_c.py) (Apache License 2.0)
