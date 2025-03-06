# Search Pioneer Ranking CLI

[![NuGet Release][CLI-image]][CLI-nuget-url]

[CLI-nuget-url]:https://www.nuget.org/packages/SearchPioneer.Ranking.Cli/
[CLI-image]:
https://img.shields.io/nuget/v/SearchPioneer.Ranking.Cli.svg

A Learning to Rank (LTR) command line interface (CLI) tool for training rankers with
[LightGBM](https://lightgbm.readthedocs.io/en/stable/) and FastTree.

## What is Learning to Rank (LTR)?

Learning to Rank (LTR) is a technique in machine learning that trains models to optimize the 
ranking order of items in a list based on relevance to a specific query or user intent.
The goal is to improve the quality of search results, recommendations, and other ranked lists by
understanding and modeling what users find most relevant or useful. LTR is widely used in search engines,
recommendation systems, and information retrieval to enhance user satisfaction and engagement.

## Installation

The CLI uses .NET 9, so ensure your system has
[.NET Runtime installed](https://dotnet.microsoft.com/en-us/download/dotnet/9.0).

To add as a global .NET command line tool

```sh
dotnet tool install -g SearchPioneer.Ranking.Cli --prerelease
```

### Tab completion

Tab completion can be enabled for the ranking CLI by following the
[System.CommandLine instructions](https://learn.microsoft.com/en-us/dotnet/standard/commandline/tab-completion#enable-tab-completion):

1. Install the [dotnet-suggest](https://nuget.org/packages/dotnet-suggest) global tool.
2. Add the appropriate shim script to your shell profile. You may have to create a shell profile file.
   The shim script forwards completion requests from your shell to the dotnet-suggest tool,
   which delegates to the appropriate ranking CLI app.

    1. For bash, add the contents of [dotnet-suggest-shim.bash](https://github.com/dotnet/command-line-api/blob/main/src/System.CommandLine.Suggest/dotnet-suggest-shim.bash) to `~/.bash_profile`.
    2. For zsh, add the contents of [dotnet-suggest-shim.zsh](https://github.com/dotnet/command-line-api/blob/main/src/System.CommandLine.Suggest/dotnet-suggest-shim.zsh) to `~/.zshrc`.
    3. For PowerShell, add the contents of [dotnet-suggest-shim.ps1](https://github.com/dotnet/command-line-api/blob/main/src/System.CommandLine.Suggest/dotnet-suggest-shim.ps1)
       to your PowerShell profile. You can find the expected path to your PowerShell profile by running the following command in your console:

        ```powershell
        echo $PROFILE
        ```

## Usage

To see all the commands supported by the command line tool

```sh
dotnet-ranking --help
```

An outline of the main commands follows.

### Train command

Trains rankers with LightGBM or FastTree. To see the available
command line options

```sh
dotnet-ranking train --help
```

At a minimum, a training data set and a test data set are provided in 
[LETOR / SVM-Rank format](https://searchpioneer.github.io/ranklib-dotnet/documentation/file-formats/training-file-format.html).
Each row in the data set contains

- the relevance label, which is typically a value in the range `[0, 1, 2, 3, 4]` where `0` is not relevant, 
  and `4` is perfect relevance, or in the range `[0, 1]` where `0` is not relevant and `1` is relevant.
- the id of the query.
- a list of features and their values, in ascending order.
- optional comments for the row. These are typically the document ID and query text.

```sh
dotnet-ranking train -t train_data.txt -e test_data.txt -m trained_model.zip
```

### Split command

Splits input training data to create train, test, and validation data sets.

A standard split is 80% training / 10% validation / 10% test

```sh
dotnet-ranking split -i input_data.txt -f 0.1 -v 0.1
```

### Fold command

Splits input training data into [K cross-validation folds](https://en.wikipedia.org/wiki/Resampling_(statistics)#Cross-validation) 
of train/test data. By default, data is split into 5 folds.

```sh
dotnet-ranking fold -i input_data.txt -o folds
```

### Transform command

Transforms a CSV data set file of features into a LETOR dataset. Allows for selection of a label, query, 
and description column, as well as the columns to use for features. All feature columns are assumed to be `float`
values.

```sh
dotnet-ranking transform -i input_data.csv -l label -q query -d description -f name_bm25 -f description_bm25 -f popularity
```
