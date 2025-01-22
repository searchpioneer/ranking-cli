using System.CommandLine;
using System.Globalization;
using System.Security.Cryptography;
using Microsoft.ML;
using Spectre.Console;

namespace SearchPioneer.Ranking.Cli;

internal class SplitCommandOptions : ICommandOptions
{
    public int? Seed { get; set; }
    public string Input { get; set; } = default!;
    public double TestFraction { get; set; }
    public double ValidationFraction { get; set; }
    
    public Dictionary<string, string?> ToDictionary() =>
        new()
        {
            { nameof(Seed), Seed?.ToString() },
            { "Input data path", Input },
            { "Test fraction", TestFraction.ToString(CultureInfo.InvariantCulture) },
            { "Validation fraction", ValidationFraction.ToString(CultureInfo.InvariantCulture) },
        };
}

internal class SplitCommand : Command<SplitCommandOptions, SplitCommandHandler>
{
    public SplitCommand()
    : base("split", "Split input training data to create train/test/validation data.")
    {
        var seedOption = new Option<int?>(["--seed", "-s"], "The seed to use for randomness. Many operations require randomness, so providing a fixed value makes operations deterministic. [default: a random value]");
        var trainOption = new Option<string>(["--input", "-i"], "The path to the input training data. The input training data must be provided in LETOR / SVM-Rank dataset format.") { IsRequired = true };
        var testFractionOption = new Option<double>(["--test-fraction", "-f"], () => 0.1, 
            "The fraction of input data to use for test data. Must be between 0 and 1 exclusive");
        testFractionOption.AddValidator(result =>
        {
            var value = result.GetValueForOption(testFractionOption);
            if (value is <= 0 or >= 1)
                result.ErrorMessage = "The test fraction must be between 0 and 1 exclusive";
        });
        
        var validationFractionOption = new Option<double>(["--validation-fraction", "-v"], () => 0, 
            "The fraction of input data to use for validation data. Must be between 0 inclusive and 1 exclusive");
        validationFractionOption.AddValidator(result =>
        {
            var value = result.GetValueForOption(validationFractionOption);
            if (value is < 0 or >= 1)
                result.ErrorMessage = "The validation fraction must be between 0 inclusive and 1 exclusive";
        });
        
        AddOption(seedOption);
        AddOption(trainOption);
        AddOption(testFractionOption);
        AddOption(validationFractionOption);
        
        AddValidator(result =>
        {
            var testFractionValue = result.GetValueForOption(testFractionOption);
            var validationFractionValue = result.GetValueForOption(validationFractionOption);

            if (testFractionValue + validationFractionValue > 1) 
                result.ErrorMessage = "The test fraction and validation fraction values must sum to less than 1";
            
            if (testFractionValue == 0 && validationFractionValue == 0) 
                result.ErrorMessage = "The test fraction, validation fraction or both must be greater than 0";
        });
    }
}

internal class SplitCommandHandler : ICommandOptionsHandler<SplitCommandOptions>
{
    private readonly IAnsiConsole _ansiConsole;

    public SplitCommandHandler(IAnsiConsole ansiConsole) => _ansiConsole = ansiConsole;

    public Task<int> HandleAsync(SplitCommandOptions options, CancellationToken cancellationToken)
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
            .Start("Splitting training data...", ctx =>
            {
                var trainTestData = mlContext.Data.TrainTestSplit(trainData, options.TestFraction + options.ValidationFraction,
                    nameof(DataPoint.QueryId), options.Seed);

                ctx.Status("Saving training data...");
                
                Save(mlContext, trainTestData.TrainSet!, options.Input, "train");
                
                if (options.TestFraction == 0)
                {
                    // we have train data and validation data.
                    ctx.Status("Saving validation data...");
                    Save(mlContext, trainTestData.TestSet!, options.Input, "validation");
                }
                else if (options.ValidationFraction == 0)
                {
                    // we have train and test data.
                    ctx.Status("Saving test data...");
                    Save(mlContext, trainTestData.TestSet!, options.Input, "test");
                }
                else
                {
                    // split again as we have train, test and validation data
                    var validationFraction = options.ValidationFraction / (options.TestFraction + options.ValidationFraction);
            
                    ctx.Status("Splitting test data...");
                    var testSplitData = mlContext.Data.TrainTestSplit(trainTestData.TestSet, validationFraction,
                        nameof(DataPoint.QueryId), options.Seed);
            
                    ctx.Status("Saving test data...");
                    Save(mlContext, testSplitData.TrainSet, options.Input, "test");
                    ctx.Status("Saving validation data...");
                    Save(mlContext, testSplitData.TestSet, options.Input, "validation");
                }
            });
        
        return Task.FromResult(0);
    }

    private static void Save(MLContext mlContext, IDataView data, string trainDataPath, string dataSetName)
    {
        var dataPoints = mlContext.Data.CreateEnumerable<DataPoint>(data, reuseRowObject: false);
        var extension = Path.GetExtension(trainDataPath);
        var path = Path.ChangeExtension(trainDataPath, dataSetName + extension);
        LetorDataFileWriter.Write(path, dataPoints);
    }
}