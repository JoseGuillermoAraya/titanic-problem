# ml_cli
This package contains a CLI to train and evaluate a ML model over the titanic problem data
Made using `poetry`

It has 3 commands:

## Train
`python cli.py train`
Options:
- `'--data-file'`, required=True, Path to the input data file
- `'--log-file'`, default='train.log', Path to the log file
- `'--model-file'`, default='model.pkl', Path to save the trained model
- `'--max_depth'`, default=3, Maximum depth of a tree
- `'--learning_rate'`, default=0.1, Learning rate
- `'--objective'`, default='binary:logistic', Objective function
- `'--eval_metric'`, default='logloss', Evaluation metric
- `'--min_child_weight'`, default=1, Minimum sum of instance weight (hessian) needed in a child
- `'--subsample'`, default=0.8, Subsample ratio of the training instances
- `'--colsample_bytree'`, default=0.8, Subsample ratio of columns when constructing each tree
- `'--min_child_weight'`, default=1, Minimum sum of instance weight (hessian) needed in a child
- `'--num_boost_round'`, default=100, Number of boosting rounds
- `'--early_stopping_rounds'`, default=10, Early stopping rounds
- `'--seed'`, default=42, Random seed
To train the model on some data

## Predict
`python cli.py predict`
Options:
- `'--input_file'`, required=True, Path to the input CSV file
- `'--output_file'`, default='predictions.csv', Path to the output CSV file
- `'--model-file'`, default='model.bin', Path to the trained model
- `'--log-file'`, default='predict.log', Path to the log file
Predict target values from input data

## Evaluate
`python cli.py evaluate`
Options:
- `'--input-file'`, required=True, Path to the input CSV file
- `'--model-file'`, required=True, Path to the trained model
- `'--output-file'`, default='evaluation_results.csv', Path to the output CSV file
- `'--log-file'`, default='evaluate.log', Path to the log file
Evaluate the model with accuracy, precission, f1, recall, auc

## Tests and coverage
`poetry run test`
Coverage will be shown in terminal and saved in html format to `./coverage-report`

## Building the package
`poetry build`
Package saves in `./dist`
