# Bottles and Cans classifier

This project is designed as a simple image classifier that specifically targets the identification and classification of bottles and cans. It leverages the capabilities of the PyTorch library, which is well-known for its robust machine learning tools. By utilizing a pre-trained model provided by PyTorch, we significantly streamline the process, avoiding the need to develop a model from scratch.

The classifier undergoes a fine-tuning process using a small, curated dataset composed exclusively of images of various types of bottles and cans. This dataset helps in adapting the generic capabilities of the pre-trained model to the specific task of distinguishing between these two categories of objects. The fine-tuning involves adjusting the model parameters slightly to better fit the peculiarities of the dataset, thereby improving accuracy and efficiency in real-world applications.

This approach not only simplifies the development of an effective classification tool but also ensures that the system is relatively lightweight and fast, making it ideal for integration into larger systems where quick, accurate classification of bottles and cans is required.

## Dataset

The dataset is from the Kaggle dataset [here](https://www.kaggle.com/datasets/moezabid/bottles-and-cans). The dataset contains images of bottles and cans, which are the two classes that the classifier is designed to distinguish between. The dataset is divided into two folders, one for each class, with each folder containing images of the respective objects. The images are of varying sizes and qualities, reflecting the diversity of real-world scenarios where the classifier might be deployed.

## Technologies Used

[![Python](./asset/Python.png)](https://twitter.com/sawaratsuki1004)

## Installation

...

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.