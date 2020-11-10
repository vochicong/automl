set -ex
shopt -s globstar

autoflake --in-place --remove-all-unused-imports --remove-unused-variables \
    **/*.py

isort --atomic -m 3 .

black .
