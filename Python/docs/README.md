Documentation
-------------

To contribute, edit the *.md files within the docs folder. Then execute the following from within the easyml/Python/docs folder:

```bash
sphinx-build . _build
```

To autobuild and see changes in real time:

```bash
sphinx-autobuild . _build
```

Then go to http://127.0.0.1:8000/.

To run notebooks, execute the following from the `./Python/docs/vignettes` directory:

```bash
jupyter notebook
```

To build vignettes, execute the following from the `./Python/docs/vignettes` directory:

```bash
jupyter nbconvert cocaine.ipynb --to markdown
jupyter nbconvert prostate.ipynb --to markdown
jupyter nbconvert titanic.ipynb --to markdown
```

Workflow for updating documenation:

1) Make any changes to the Python docstrings.
2) Then execute the following from the `./Python` directory:

```bash
pip uninstall easymlpy -y
python setup.py install
sphinx-build ./docs ./docs/_build
sphinx-autobuild ./docs ./docs/_build
```

Then go to http://127.0.0.1:8000/ to view the documentation.
