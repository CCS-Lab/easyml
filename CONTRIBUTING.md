Thanks for taking the plunge!

## Reporting Issues

* It's always good to start with a quick search for an existing issue to post on, or related issues for context, before opening a new issue.
* Including minimal examples is greatly appeciated.
* If it's a bug, or unexpected behaviour, reproducing on the latest development version is a good gut check and can streamline the process.
* It's always helpful to include information on the OS, language version, package version, dependency versions, and any other information that might be relevant. For R, please include the output of `sessionInfo()`. 

## Contributing

* Feel free to open, or comment on, an issue and solicit feedback early on, especially if you're unsure about aligning with design goals and direction, or if relevant historical comments are ambiguous.
* Pair new functionality with tests, and bug fixes with tests that fail pre-fix. Increasing test coverage as you go is always nice.
* Aim for atomic commits, if possible, e.g. `change 'foo' behavior like so` & `'bar' handles such and such corner case`, rather than `update 'foo' and 'bar'` & `fix typo` & `fix 'bar' better`.
* Pull requests are tested against release and development branches of R and Python.
* The style guidelines outlined below are not the personal style of most contributors, but for consistency throughout the project, we've adopted them.

## Style Guidelines

* For R, please follow [Hadley's style guide](http://adv-r.had.co.nz/Style.html).
* For Python, please follow the [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/).
