# TL;DR
Checks dependencies in `pom.xml` and `build.gradle` files


## Running and options
To execute you need to first set an environment variable `GITHUB_TOKEN`

* set this to a valid token with permissions to access (read) the organisations you want to scan

There are two supported options:
* `--dep_file` provide a previously generated file (default name is `output.json`)
* `--plot` provide a setting for the type of plot to draw with the data (default is a histogram)

## Limitations
Currently cannot search through monorepos which contain multiple projects and associated build files at arbitrary paths due to limits of the GitHub search api (hit rate limits very quickly when searching)