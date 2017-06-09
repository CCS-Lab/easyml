Documentation
-------------

To contribute, edit the *.rst files within the docs folder. Then execute the following from within the easyml/Python/docs folder:

```bash
sphinx-build . _build
```

To autobuild and see changes in real time:

```bash
sphinx-autobuild . _build
```

Then go to http://127.0.0.1:8000/.

To automatically generate documentation from the docstrings:

```bash
sphinx-autobuild . _build
```
