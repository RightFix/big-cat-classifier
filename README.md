# Big Cat Classifier

A deep learning image classification app that identifies big cats (Leopard and Tiger) from uploaded images. Built with a custom CNN trained on TensorFlow/Keras and deployed via Streamlit.

GET 324 — Group C10 | University of Uyo

---

## Demo

Upload any image of a lion or tiger and the model returns the predicted class with confidence scores for each category.

---

## Repository Structure

```
big-cat-classifier/
├── models/
│   └── tl_feature_extraction_best.keras   # saved trained model
├── app.py                                  # Streamlit web app
├── big-cat-classifier-notebook            # training notebook
├── requirements.txt                        # Python dependencies
├── .python-version                         # Python 3.11
└── README.md
```

---

## Model

- Architecture: Custom CNN with residual skip connections (4 convolutional blocks)
- Input shape: `(224, 224, 3)`
- Classes: `african leopard`, `tiger`, ` clouded leopard`, `snow leopard` 
- Loss: `sparse_categorical_crossentropy`
- Optimizer: `Adam`
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

---

## Dataset

[Big Cats Image Classification Dataset — Kaggle](https://www.kaggle.com/datasetsgpiosenka/cats-in-the-wild-image-classification)

```python
import kagglehub

path = kagglehub.dataset_download("gpiosenka/cats-in-the-wild-image-classification")
print("Path to dataset files:", path)
```

## Run Locally

1. Clone the repo

```bash
git clone https://github.com/RightFix/big-cat-classifier.git
cd big-cat-classifier
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app

```bash
streamlit run app.py
```

---

## Tech Stack

- Python 3.11
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow

---

## Contributors

- [Righteousness Ude](https://github.com/RightFix)
- [Nmesoma Victory](https://github.com/Nmeso1n) 
- [Nsikan Ebong](https://github.com/nsikanebong) 
- [Godspower Okon](https://github.com/Aidenstar17)