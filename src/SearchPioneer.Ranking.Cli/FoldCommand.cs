using System.CommandLine;
using System.Security.Cryptography;
using Microsoft.ML;
using Spectre.Console;

namespace SearchPioneer.Ranking.Cli;

internal class FoldCommandOptions : ICommandOptions
{
    public int? Seed { get; set; }
    public string Input { get; set; } = default!;
    public int Folds { get; set; }
    public string? OutputDir { get; set; }
    
    public Dictionary<string, string?> ToDictionary() =>
        new()
        {
            { nameof(Seed), Seed?.ToString() },
            { "Input data path", Input },
            { "Number of folds", Folds.ToString() },
            { "Output Directory", OutputDir },
        };
}

internal class FoldCommand : Command<FoldCommandOptions, FoldCommandOptionsHandler>
{
    public FoldCommand() 
        : base("fold", "Split input training data into K cross-validation folds of train/test data.")
    {
        var seedOption = new Option<int?>(["--seed", "-s"], "The seed to use for randomness. Many operations require randomness, so providing a fixed value makes operations deterministic. [default: a random value]");
        var trainOption = new Option<string>(["--input", "-i"], "The path to the input training data. The input training data must be provided in LETOR / SVM-Rank dataset format.") { IsRequired = true };
        var foldsOption = new Option<int>(["--folds", "-f"], () => 5, 
            "The number of cross-validation folds to split data into. Must be greater than 1");
        
        foldsOption.AddValidator(result =>
        {
            var folds = result.GetValueForOption(foldsOption);
            if (folds <= 1)
                result.ErrorMessage = "The number of cross-validation folds must be greater than 1.";
        });
        
        var outputDirOption = new Option<string?>(["--output-dir", "-o"], 
            "The output directory for folds of train/test data. If unspecified, the current directory will be used.");
        
        AddOption(seedOption);
        AddOption(trainOption);
        AddOption(foldsOption);
        AddOption(outputDirOption);
    }
}

internal class FoldCommandOptionsHandler : ICommandOptionsHandler<FoldCommandOptions>
{
    private readonly IAnsiConsole _ansiConsole;

    public FoldCommandOptionsHandler(IAnsiConsole ansiConsole) => _ansiConsole = ansiConsole;

    public Task<int> HandleAsync(FoldCommandOptions options, CancellationToken cancellationToken)
    {
        // set the seed here rather than in options default value, so that --help isn't confusing by
        // including a random value.
        options.Seed ??= RandomNumberGenerator.GetInt32(int.MaxValue);
        
        var mlContext = new MLContext(options.Seed);
        var optionsTable = new Table()
            .Title("[bold]Command Options[/]")
            .Border(TableBorder.Rounded)
            .AddColumn("Option")
            .AddColumn("Value");

        foreach (var kvp in options.ToDictionary()) 
            optionsTable.AddRow(kvp.Key, kvp.Value ?? "[grey]null[/]");
        
        _ansiConsole.Write(optionsTable);
        
        IDataView trainData = null!;
        _ansiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .Start("Loading training data...", _ =>
            {
                trainData = mlContext.Data.LoadFromEnumerable(LetorDataFileReader.Read(options.Input));
            });
        
        _ansiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .Start($"Splitting training data into {options.Folds} folds...", ctx =>
            {
                var trainTestDataFolds = 
                    mlContext.Data.CrossValidationSplit(trainData, options.Folds, nameof(DataPoint.QueryId), options.Seed);

                if (options.OutputDir != null) 
                    Directory.CreateDirectory(options.OutputDir);
                
                for (var i = 0; i < trainTestDataFolds.Count; i++)
                {
                    var foldNumber = i + 1;
                    var outputDirectory = options.OutputDir != null 
                        ? Path.Combine(options.OutputDir, $"fold{foldNumber}") 
                        : $"fold{foldNumber}";
                    
                    Directory.CreateDirectory(outputDirectory);
                    ctx.Status($"Saving fold {foldNumber} to {outputDirectory}...");
                    var foldData = trainTestDataFolds[i];
                    Save(mlContext, foldData.TrainSet, outputDirectory, "train");
                    Save(mlContext, foldData.TestSet, outputDirectory, "test");
                }
            });
        
        return Task.FromResult(0);
    }
    
    private static void Save(MLContext mlContext, IDataView data, string outputDirectory, string dataSetName)
    {
        var dataPoints = mlContext.Data.CreateEnumerable<DataPoint>(data, reuseRowObject: false);
        var path = Path.Combine(outputDirectory, $"{dataSetName}.txt");
        LetorDataFileWriter.Write(path, dataPoints);
    }
}