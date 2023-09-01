# Decentralization

Platform Decentralization analysis


# Installing dependencies

You need a recent version of Python. And you need [Poetry](https://python-poetry.org/docs/). Poetry would typically be installed with:

```
curl -sSL https://install.python-poetry.org | python3 -
```
But check the official docs if the above does not work on your platform.

After that, install all deps:
```
poetry install
```

And you're ready to rock.

# Usage

Demo, that pulls that latest list of subnets and nodes from the Public Dashboard API, and builds two Pandas DataFrames:

```
poetry run python3 ic_topology/main.py
```
