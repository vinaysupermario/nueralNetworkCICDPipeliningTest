# How to run locally:

## Create a virtual environment and activate it:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Install dependencies:
```
pip install -r requirements.txt
```

## Train the model:
```
python train.py --config config.yaml
```

## Run tests:
```
pytest test_model.py -v
```

## To push to GitHub:

1. Create a new repository on GitHub
2. Initialize git and push the code:
```
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```
