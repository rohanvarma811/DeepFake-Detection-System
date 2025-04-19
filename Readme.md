# DeepFake Detection System

DeepFake Detection Web-App 🖥 using Deep Learning(ResNext and LSTM), Flask and ReactJs where you can predict whether a video is FAKE Or REAL along with the confidence ratio.

## Table of Contents

1. [Installation](#installation)
2. [Technologies Used](#technologies-used)
3. [Running the App](#running-the-app)
4. [Results](#Results)
5. [License](#license)

## Installation

### Prerequisites

- Python
- React
- npm
- Flask

### Cloning the Repository

```bash
git clone https://github.com/rohanvarma811/mytasks.git
cd DeepFake-Detection
pip install -r requirements.txt
python server.py
```

### Installing Dependencies

- Read the requirements.txt file for Installing Dependencies
- You may require to download the CMake Software

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Python, Flask
- **Machine Learning Framework:**: dlib, PyTorch (or TensorFlow)
- **Model:**: Deep Learning model (df_model.pt)
- **File Handling:**:  OpenCV, NumPy
- **Web Server:**: python-dotenv
- **Web Framework:**: Flask

## Running the App

1. Make sure CMake is Downloaded and added to Path.
2. Go to the DeepFake-Detection folder:

```bash
cd DeepFake-Detection
```

3. Start the app:

```bash
python server.py
```

4. Open your browser and navigate to `http://127.0.0.1:5000` to see the app.

## Note
1. In the root folder(DeepFake_Detection), create a new folder called "Uploaded_Files".

2. In the root folder(DeepFake_Detection), create a new folder called "model" and add the [model file](https://drive.google.com/drive/folders/1-zErGZ9T89TplQs3ws4QVRFlqE-ljW6l?usp=sharing) in it.

<b>Add these folders to the root folder(DeepFake-Detection). Since, the path has already been given to the "server.py" file and also to avoid any path related errors.</b>

## Results

1) Accuracy of the Model:
<img width="250" height="50" alt="Model Accuracy" src="https://user-images.githubusercontent.com/58872872/133935912-1def7615-6538-4c88-9134-8f94a9367965.png">

2) Training and Validation Accuracy Graph:
<img width="378" alt="Accuracy Graph" src="https://user-images.githubusercontent.com/58872872/133936040-4bfa44a7-45c5-499b-8a10-f253cbcab56c.png">

3) Training and Validation Loss Graph:
<img width="381" alt="Loss Graph" src="https://user-images.githubusercontent.com/58872872/133935983-b4d9275f-e841-4b69-86cd-79c770ea2aa1.png">

4) Confusion Matrix:
<img width="402" alt="Confusion Matrix" src="https://user-images.githubusercontent.com/58872872/133936080-d2b39804-4a99-47b8-8be4-87ba77161961.png">

## License

This project is licensed under the MIT License.