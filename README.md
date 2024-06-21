# CTR_stacking

The following codes are the solutions **(6st place, private score: 0.78882)** for the dacon competition.

# 1. Environmental settings
## 1.1 Clone this repository
```
git clone https://github.com/GNOEYHEAT/CTR_stacking.git
cd CTR_stacking
```
## 1.2 Install packages
```
pip install -r requirements.txt 
```
## 1.3 Directory Structure
```
|-- README.md
|-- data
|   |-- data.zip
|   |-- pp_test_ce.parquet
|   |-- pp_train_ce.parquet
|   |-- sample_submission.csv
|   |-- test.csv
|   |-- train.csv
|-- requirements.txt
|-- src
|   |-- Preprocess.py
|   |-- config.py
|   |-- model.py
|   `-- run.py
```
# 2. Data Preprocessing
```
cd ./src
python Preprocess.py
```
# 3. Train Model
```
python run.py
```
# 4. Submission
The final submission is submission/stack.csv
