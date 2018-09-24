# Sensei

Project sensei is developed for course Machine Learning at faculty of Mathematics, University of Belgrade.
Since it's written in C++ it's only purpose is to enable the author to get in-depth understanding of
underlaying concept of Restricted Boltzmann Machines. Beside RBM, sensei provides framework for deep learning
so that further development is still open.

## Dependencies
* armadillo

## Usage
From root folder execute build process
```bash
make
```
In order to run example for training RBM on MNIST dataset, first you need a dataset. I've prepaired
MNIST dataset in CSV format that's readable by sensei (although, any CSV should be readable). This
file should be placed inside `test/dataset` folder. One can download the dataset from [here](https://drive.google.com/file/d/12Vx-E484RyFWLndMq7QqP7cAy4TT8pTb/view?usp=sharing)
```
After the dataset has been downloaded, issue this command from the project root to train RBM model:
```bash
make run
```

## Authors
* **Uros Stegic** - *urosstegic@gmx.com* - [uros-stegic](https://github.com/uros-stegic)

## License
This project is licensed under the GNU GPL-3 License - see the [LICENSE](LICENSE) file for details

