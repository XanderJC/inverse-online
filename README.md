
# [Inverse Online Learning: Understanding Non-Stationary and Reactionary Policies](https://openreview.net/forum?id=DYypjaRdph2)

### Alex J. Chan, Alicia Curth, and Mihaela van der Schaar

### International Conference on Learning Representations (ICLR) 2022
 [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
 <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>


Last Updated: 14 February 2022

Code Author: Alex J. Chan (ajc340@cam.ac.uk)

This repo contains a PyTorch example implementation of the inverse online learning algorithm presented in our paper. The code is ready to run on a synthetic data example and should be simple to apply to arbitrary datasets.

This repo is pip installable - clone it, optionally create a virtual env, and install it (this will automatically install dependencies):

```shell
git clone https://github.com/XanderJC/inverse-online.git

cd inverse-online

pip install -e .
```


Example usage:

```python
from iol.models import AdaptiveLinearModel
from iol.data_loading import generate_linear_dataset
from iol.constants import HYPERPARAMS

model = AdaptiveLinearModel(**HYPERPARAMS)

training_data   = generate_linear_dataset(10000, 50, seed=41310)
validation_data = generate_linear_dataset(1000,  50, seed=41311).get_whole_batch()
test_data       = generate_linear_dataset(1000,  50, seed=41312).get_whole_batch()

model.fit(
    training_data,
    batch_size=100,
    epochs=5,
    learning_rate=0.01,
    validation_set=validation_data,
)

print(model.validation(test_data))
```

This example can be run simply from the shell using:

```shell
python iol/demo.py
```


### Citing 

If you use this software please cite as follows:

```
@inproceedings{chan2022inverse,
title={Inverse Online Learning: Understanding Non-Stationary and Reactionary Policies},
author={Alex James Chan and Alicia Curth and Mihaela van der Schaar},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=DYypjaRdph2}
}
```
