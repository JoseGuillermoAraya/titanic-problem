#  Titanic Problem

- Install the project with `poetry install`
- use one of the 4 commands to interact with the model
    - `ml_cli train --data-file=./data/raw/train.csv --model-file=./data/models/model`
    - `ml_cli evaluate --input-file=./data/raw/train.csv --model-file=./data/models/model --output-file=./data/evaluations/eval.csv`
    - `ml_cli predict --input_file=./data/raw/test.csv --output_file=./data/predictions/prediction.csv --model-file=./data/models/model`
    - `ml_cli get-feature-importance --model-file=./data/models/model --ouput-file=./data/feature_importances/feature_importance.csv`