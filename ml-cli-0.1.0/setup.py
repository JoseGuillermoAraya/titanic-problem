# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['ml_cli', 'ml_cli.commands', 'ml_cli.data', 'ml_cli.models', 'ml_cli.utils']

package_data = \
{'': ['*']}

install_requires = \
['click>=8.1.3,<9.0.0',
 'joblib>=1.2.0,<2.0.0',
 'logging>=0.4.9.6,<0.5.0.0',
 'pandas>=1.5.3,<2.0.0',
 'scikit-learn>=1.2.1,<2.0.0',
 'xgboost>=1.7.4,<2.0.0']

entry_points = \
{'console_scripts': ['ml_cli = ml_cli.cli:cli', 'test = scripts:test']}

setup_kwargs = {
    'name': 'ml-cli',
    'version': '0.1.0',
    'description': 'Simple CLI to train, evaluate and predict with a XGB model on the Titanic Problem',
    'long_description': "# ml_cli\nThis package contains a CLI to train and evaluate a ML model over the titanic problem data\nMade using `poetry`\n\nIt has 3 commands:\n\n## Train\n`python cli.py train`\nOptions:\n- `'--data-file'`, required=True, Path to the input data file\n- `'--log-file'`, default='train.log', Path to the log file\n- `'--model-file'`, default='model.pkl', Path to save the trained model\n- `'--max_depth'`, default=3, Maximum depth of a tree\n- `'--learning_rate'`, default=0.1, Learning rate\n- `'--objective'`, default='binary:logistic', Objective function\n- `'--eval_metric'`, default='logloss', Evaluation metric\n- `'--min_child_weight'`, default=1, Minimum sum of instance weight (hessian) needed in a child\n- `'--subsample'`, default=0.8, Subsample ratio of the training instances\n- `'--colsample_bytree'`, default=0.8, Subsample ratio of columns when constructing each tree\n- `'--min_child_weight'`, default=1, Minimum sum of instance weight (hessian) needed in a child\n- `'--num_boost_round'`, default=100, Number of boosting rounds\n- `'--early_stopping_rounds'`, default=10, Early stopping rounds\n- `'--seed'`, default=42, Random seed\nTo train the model on some data\n\n## Predict\n`python cli.py predict`\nOptions:\n- `'--input_file'`, required=True, Path to the input CSV file\n- `'--output_file'`, default='predictions.csv', Path to the output CSV file\n- `'--model-file'`, default='model.bin', Path to the trained model\n- `'--log-file'`, default='predict.log', Path to the log file\nPredict target values from input data\n\n## Evaluate\n`python cli.py evaluate`\nOptions:\n- `'--input-file'`, required=True, Path to the input CSV file\n- `'--model-file'`, required=True, Path to the trained model\n- `'--output-file'`, default='evaluation_results.csv', Path to the output CSV file\n- `'--log-file'`, default='evaluate.log', Path to the log file\nEvaluate the model with accuracy, precission, f1, recall, auc\n\n## Tests and coverage\n`poetry run test`\nCoverage will be shown in terminal and saved in html format to `./coverage-report`\n\n## Building the package\n`poetry build`\nPackage saves in `./dist`\n",
    'author': 'JoseGuillermoAraya',
    'author_email': 'jose.araya@bci.cl',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.10,<4.0',
}


setup(**setup_kwargs)
