# Code for the paper

**Ensuring Reliability in Programming Knowledge Tracing:  
A Re-evaluation of Attention-augmented Models and Experimental Protocols**

Accepted at the International Conference on Intelligent Tutoring Systems (ITS 2026).  
To appear in Springer.

## Setup
- Please note that this runs on GPUs
- Running a Docker container using the Dockerfile-pytorch2.7.1-cuda12.8 is better to setup the environment.

### Install the required dependencies
```
pip install -r requirements.txt
```

## Experiments

### Dataset

We use the CodeWorkout dataset (Price and Shi, 2021), available from the PSLC DataShop:

https://pslcdatashop.web.cmu.edu/Files?datasetId=3458

Access to the dataset may be restricted. Please request access through DataShop if it is not publicly downloadable.

- download and move or create link of dataset folder under project directory: the folder name must be ./dataset/CodeWorkout

### Ready to experiment
- split student ids to train and test
```
cd exps
python split_student_ids.py
```

### Code-DKT original
#### Attribution
This repository includes code adapted from the following open-source project:
- [Code-DKT](https://github.com/YangAzure/Code-DKT), licensed under the MIT License.

#### Move into the directory
```
cd exps/Code-DKT-org
```

#### Make code AST paths from java code
```
python path_extractor.py
```

#### Preprocess
```
python preprocessing.py
```

#### Experiment
```
sh _run_all.sh
```

### CodeDKT_ECKT
#### Move into the directory
```
cd exps/CodeDKT_ECKT
```
#### Make code AST paths from java code
```
sh _preprocess_code_ast_path.sh
```
#### Make problem difficulty and save Knowledge KCs from student's code for ECKT
```
sh _preprocess_problem_difficulty.sh

sh _preprocess_knowledge.sh
```
#### Preprocess for CodeDKT, ECKT.
```
sh _preprocess_train_test.sh
```

#### Experiment
```
sh _grid_all_train-msl_50.sh
```

#### Experiment with maximum sequence length set to 100
```
sh _grid_all_train-msl_100.sh

```
