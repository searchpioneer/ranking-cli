using System.CommandLine;
using System.Security.Cryptography;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;
using Spectre.Console;

namespace SearchPioneer.Ranking.Cli;

internal class TrainCommandOptions : ICommandOptions
{
    public Implementation Implementation { get; set; }
    public int? Seed { get; set; }
    public string Train { get; set; } = default!;
    public string Test { get; set; } = default!;
    public string? Validation { get; set; } = default!;
    public string? Model { get; set; } = default!;
    public int? Leaves { get; set; }
    public int? MinExamples { get; set; }
    public double? LearningRate { get; set; }
    public int Iterations { get; set; }
    public int DcgTruncationLevel { get; set; }
    public bool FeatureStats { get; set; }
    public int Permutations { get; set; }
    
    public Dictionary<string, string?> ToDictionary() =>
        new()
        {
            { nameof(Cli.Implementation), Implementation.ToString() },
            { nameof(Seed), Seed?.ToString() },
            { "Train data path", Train },
            { "Test data path", Test },
            { "Validation data path", Validation },
            { "Model output path", Model },
            { "Number of leaves", Leaves?.ToString() },
            { "Minimum examples", MinExamples?.ToString() },
            { "Learning rate", LearningRate?.ToString() },
            { "Number of iterations", Iterations.ToString() },
            { "(N)DCG truncation level", DcgTruncationLevel.ToString() },
            { "Calculate feature importance statistics", FeatureStats ? "true" : "false" },
            { "Permutation count", Permutations.ToString() }
        };
}

internal class TrainCommand : Command<TrainCommandOptions, TrainCommandHandler>
{
    public TrainCommand()
        : base("train", "Train a ranker with LightGBM or FastTree.")
    {
        var seedOption = new Option<int?>(["--seed", "-s"], "The seed to use for randomness. Many operations require randomness, so providing a fixed value makes operations deterministic. [default: a random value]");
        var implementationOption = new Option<Implementation>(["--implementation", "-a"], () => Implementation.LightGbm, 
            "The ranking algorithm implementation");
        var trainOption = new Option<string>(["--train", "-t"], "The path to the training data. The input training data must be provided in LETOR / SVM-Rank dataset format.") { IsRequired = true };
        var testOption = new Option<string>(["--test", "-e"], "The path to the test data. The input training data must be provided in LETOR / SVM-Rank dataset format.") { IsRequired = true };
        var validationOption = new Option<string?>(["--validation", "-v"], "The path to the validation data. The input training data must be provided in LETOR / SVM-Rank dataset format.");
        var modelOption = new Option<string>(["--model", "-m"], "The path to save the trained model zip to");
        var leavesOption = new Option<int?>(["--leaves", "-l"], "The maximum number of leaves in one tree.");
        var minExamplesOption = new Option<int?>(["--min-examples", "-n"], 
            "The minimal number of data points required to form a new tree leaf.");
        var learningRateOption = new Option<double?>(["--learning-rate", "-r"], "The learning rate.");
        var iterationsOption = new Option<int>(["--iterations", "-i"], () => 100, 
            "The number of boosting iterations. A new tree is created in each iteration, so this is equivalent to the number of trees.");
        var dcgTruncationOption = new Option<int>(["--dcg-truncation-level", "-d"], () => 10, 
            "Maximum truncation level for computing (N)DCG.");
        var featureStatsOption = new Option<bool>(["--feature-stats", "-f"], () => true, 
            "Calculate feature importance statistics using Permutation Feature Importance (PFI). " +
            "The output statistics are ordered from most important to least important feature. PFI may take some time on large datasets with many features.");
        var permutationCountOption = new Option<int>(["--permutations", "-p"], () => 1, 
            "Number of permutations to use when calculating Permutation Feature Importance (PFI).");
        
        AddOption(seedOption);
        AddOption(implementationOption);
        AddOption(trainOption);
        AddOption(testOption);
        AddOption(validationOption);
        AddOption(modelOption);
        AddOption(leavesOption);
        AddOption(minExamplesOption);
        AddOption(learningRateOption);
        AddOption(iterationsOption);
        AddOption(dcgTruncationOption);
        AddOption(featureStatsOption);
        AddOption(permutationCountOption);
    }
}

internal class TrainCommandHandler : ICommandOptionsHandler<TrainCommandOptions>
{
    private readonly IAnsiConsole _ansiConsole;

    public TrainCommandHandler(IAnsiConsole ansiConsole) => _ansiConsole = ansiConsole;

    public Task<int> HandleAsync(TrainCommandOptions options, CancellationToken cancellationToken)
    {
        int featuresCount;
        using (var stream = File.OpenRead(options.Train))
        {
            using var streamReader = new StreamReader(stream);
            var line = streamReader.ReadLine();
            if (string.IsNullOrWhiteSpace(line))
            {
                _ansiConsole.MarkupLine("[red]Training data is empty.[/]");
                return Task.FromResult(1);
            }

            if (!DataPoint.TryParse(line, out var dataPoint))
            {
                _ansiConsole.MarkupLine($"[red]Not a valid data point: {line}[/]");
                return Task.FromResult(1);
            }

            featuresCount = dataPoint!.Features.Length;
        }

        // set the seed here rather than in options default value, so that --help isn't confusing by
        // including a random value.
        options.Seed ??= RandomNumberGenerator.GetInt32(int.MaxValue);

        var optionsTable = new Table()
            .Title("[bold]Command Options[/]")
            .Border(TableBorder.Rounded)
            .AddColumn("Option")
            .AddColumn("Value");

        foreach (var kvp in options.ToDictionary()) 
            optionsTable.AddRow(kvp.Key, kvp.Value ?? "[grey]null[/]");
        
        optionsTable.AddRow("Features count", featuresCount.ToString());
        _ansiConsole.Write(optionsTable);

        var mlContext = new MLContext(options.Seed);
        var schemaDefinition = SchemaDefinition.Create(typeof(DataPoint));
        var featuresColumn = schemaDefinition[nameof(DataPoint.Features)];
        featuresColumn.ColumnType = new VectorDataViewType(NumberDataViewType.Single, featuresCount);

        cancellationToken.ThrowIfCancellationRequested();

        IDataView trainData = null!;
        _ansiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .Start("Loading training data...", _ =>
            {
                trainData = mlContext.Data.LoadFromEnumerable(LetorDataFileReader.Read(options.Train), schemaDefinition);
            });

        var dataPipeline = CreateDataPipeline(mlContext);
        IEstimator<ITransformer> rankingPipeline = options.Implementation == Implementation.LightGbm 
            ? CreateLightGbmPipeline(mlContext, dataPipeline, options)
            : CreateFastTreePipeline(mlContext, dataPipeline, options);

        cancellationToken.ThrowIfCancellationRequested();

        // Train the model
        ITransformer model = null!;
        _ansiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .Start("Training the model on the training data...", _ =>
            {
                model = rankingPipeline.Fit(trainData);
            });

        cancellationToken.ThrowIfCancellationRequested();

        if (options.FeatureStats)
        {
            _ansiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .Start("Calculate feature importance on training data...", _ =>
                {
                    switch (options.Implementation)
                    {
                        case Implementation.LightGbm:
                            CalculateFeatureImportance<LightGbmRankingModelParameters>(mlContext, model, trainData,
                                options);
                            break;
                        case Implementation.FastTree:
                            CalculateFeatureImportance<FastTreeRankingModelParameters>(mlContext, model, trainData,
                                options);
                            break;
                    }
                });
            
            cancellationToken.ThrowIfCancellationRequested();
        }

        // Evaluate on validation data if provided
        if (!string.IsNullOrEmpty(options.Validation))
        {
            _ansiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .Start("Loading and evaluating on the validation data...", _ =>
                {
                    var validationData = mlContext.Data.LoadFromEnumerable(LetorDataFileReader.Read(options.Validation), schemaDefinition);
                    EvaluateModel(mlContext, model, validationData, options, "Validation");
                });
        }
        
        _ansiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .Start("Loading and evaluating on the test data...", _ =>
            {
                var testData = mlContext.Data.LoadFromEnumerable(LetorDataFileReader.Read(options.Test), schemaDefinition);
                EvaluateModel(mlContext, model, testData, options, "Test");
            });

        cancellationToken.ThrowIfCancellationRequested();

        if (!string.IsNullOrEmpty(options.Model))
        {
            _ansiConsole.Status()
                .Spinner(Spinner.Known.Dots)
                .Start("Saving the model...", _ => mlContext.Model.Save(model, trainData.Schema, options.Model));
        }

        cancellationToken.ThrowIfCancellationRequested();

        return Task.FromResult(0);
    }

    private void EvaluateModel(MLContext mlContext, ITransformer model, IDataView data, TrainCommandOptions options, string datasetName)
    {
        var predictions = model.Transform(data);
        var evaluatorOptions = new RankingEvaluatorOptions
        {
            DcgTruncationLevel = options.DcgTruncationLevel
        };

        var metrics = mlContext.Ranking.Evaluate(predictions, evaluatorOptions, rowGroupColumnName: nameof(Prediction.QueryId));
        
        var table = new Table
        {
            Title = new TableTitle($"{datasetName} Data Evaluation Metrics")
                .SetStyle(new Style(decoration: Decoration.Bold))
        };
        table.AddColumn("Metric");
        for (var i = 1; i <= options.DcgTruncationLevel; i++)
        {
            table.AddColumn($"@{i}");
        }
        
        var dcgRow = new List<string> { "DCG" };
        dcgRow.AddRange(metrics.DiscountedCumulativeGains.Select(d => d.ToString("F4")));
        table.AddRow(dcgRow.ToArray());
        
        var ndcgRow = new List<string> { "NDCG" };
        ndcgRow.AddRange(metrics.NormalizedDiscountedCumulativeGains.Select(d => d.ToString("F4")));
        table.AddRow(ndcgRow.ToArray());

        _ansiConsole.Write(table);
        _ansiConsole.WriteLine();
    }

    private void CalculateFeatureImportance<TModel>(MLContext mlContext, ITransformer model, IDataView data, TrainCommandOptions options) 
        where TModel : class
    {
        if (model is TransformerChain<RankingPredictionTransformer<TModel>> rankingPipeline)
        {
            var predictions = model.Transform(data);
            var statistics = 
                mlContext.Ranking.PermutationFeatureImportance(
                    rankingPipeline.LastTransformer, 
                    predictions, 
                    rowGroupColumnName: nameof(DataPoint.QueryId),
                    permutationCount: options.Permutations);

            var sortedFeatureIndices = statistics
                .Select((metric, index) => new { index, metric })
                .OrderByDescending(stat => Math.Abs(stat.metric.NormalizedDiscountedCumulativeGains[^1].Mean))
                .Select(stat => stat.index);

            var statsTable = new Table
            {
                Title = new TableTitle("Train Feature Importance Statistics")
                    .SetStyle(new Style(decoration: Decoration.Bold))
            };
            statsTable.AddColumn("Feature");
            statsTable.AddColumn($"DCG@{options.DcgTruncationLevel} Change");
            statsTable.AddColumn($"DCG@{options.DcgTruncationLevel} Std Err");
            statsTable.AddColumn($"NDCG@{options.DcgTruncationLevel} Change");
            statsTable.AddColumn($"NDCG@{options.DcgTruncationLevel} Std Err");
            
            foreach (var index in sortedFeatureIndices)
            {
                var stat = statistics[index]!;
                var dcg = stat.DiscountedCumulativeGains[^1];
                var ndcg = stat.NormalizedDiscountedCumulativeGains[^1];
                
                statsTable.AddRow(
                    // features are 1-based in LETOR datasets
                    (index + 1).ToString(), 
                    dcg.Mean.ToString("F4"), 
                    dcg.StandardError.ToString("F4"), 
                    ndcg.Mean.ToString("F4"),
                    ndcg.StandardError.ToString("F4"));
            }

            _ansiConsole.Write(statsTable);
            _ansiConsole.WriteLine();
        }
    }

    private static IEstimator<ITransformer> CreateDataPipeline(MLContext mlContext) =>
        mlContext.Transforms.Conversion.MapValueToKey(nameof(DataPoint.Label))
            .Append(mlContext.Transforms.Conversion.Hash(nameof(DataPoint.QueryId), numberOfBits: 20));

    private static EstimatorChain<RankingPredictionTransformer<LightGbmRankingModelParameters>> CreateLightGbmPipeline(
        MLContext mlContext, 
        IEstimator<ITransformer> dataPipeline, 
        TrainCommandOptions options)
    {
        var lightGbmRankingTrainer = mlContext.Ranking.Trainers.LightGbm(
            labelColumnName: nameof(DataPoint.Label),
            featureColumnName: nameof(DataPoint.Features),
            rowGroupColumnName: nameof(DataPoint.QueryId),
            numberOfLeaves: options.Leaves,
            minimumExampleCountPerLeaf: options.MinExamples,
            learningRate: options.LearningRate,
            numberOfIterations: options.Iterations);
        
        return dataPipeline.Append(lightGbmRankingTrainer);
    }
    
    private static EstimatorChain<RankingPredictionTransformer<FastTreeRankingModelParameters>> CreateFastTreePipeline(
        MLContext mlContext, 
        IEstimator<ITransformer> dataPipeline, 
        TrainCommandOptions options)
    {
        var fastTreeRankingTrainer = mlContext.Ranking.Trainers.FastTree(
            labelColumnName: nameof(DataPoint.Label),
            featureColumnName: nameof(DataPoint.Features),
            rowGroupColumnName: nameof(DataPoint.QueryId),
            numberOfLeaves: options.Leaves ?? 20,
            minimumExampleCountPerLeaf: options.MinExamples ?? 10,
            learningRate: options.LearningRate ?? 0.2D,
            numberOfTrees: options.Iterations);
        
        return dataPipeline.Append(fastTreeRankingTrainer);
    }
}