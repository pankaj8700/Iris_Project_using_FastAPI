# FastAPI Iris Species Prediction

This project aims to build a machine learning model using FastAPI for predicting the species of Iris flowers. The Iris dataset is a popular dataset in machine learning, consisting of measurements of four features of Iris flowers: sepal length, sepal width, petal length, and petal width.

## Installation

To get started, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/iris-project.git`
2. Navigate to the project directory: `cd iris-project`
3. Create a virtual environment: `python -m venv venv`
4. Activate the virtual environment: `source venv/bin/activate` (for Linux/Mac) or `venv\Scripts\activate` (for Windows)
5. Install the required dependencies: `pip install -r requirements.txt`

## Usage

To run the FastAPI application and make predictions, execute the following command:

```
uvicorn main:app --reload
```

Once the server is running, you can access the API documentation at `http://localhost:8000/docs` and use the provided endpoints to make predictions.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
