using System.CommandLine;
using System.Security.Cryptography;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Spectre.Console;

namespace SearchPioneer.Ranking.Cli;

internal class TransformCommandOptions : ICommandOptions
{
    public int? Seed { get; set; }
    public string Input { get; set; } = default!;
    public string? Output { get; set; } = default!;
    public char? Separator { get; set; }
    public string? Headers { get; set; }
    public string LabelColumn { get; set; } = default!;
    public string QueryIdColumn { get; set; } = default!;
    public string DescriptionColumn { get; set; } = default!;
    public List<string>? FeatureColumn { get; set; } = default!;

    public Dictionary<string, string?> ToDictionary()
    {
        return new()
        {
            { nameof(Seed), Seed?.ToString() },
            { "Input data path", Input },
            { "Output data path", Output },
            { "Column separator", Escaped(Separator) },
            { "Headers file", Headers },
            { "Label column", LabelColumn },
            { "QueryId column", QueryIdColumn },
            { "Description column", DescriptionColumn },
            { "Feature columns", FeatureColumn != null ? string.Join(",", FeatureColumn) : null },
        };
    }

    private static string? Escaped(char? c)
    {
        if (c is null)
            return null;

        return c == '\t' ? @"\t" : c.ToString();
    }
}

internal class TransformCommand : Command<TransformCommandOptions, TransformCommandHandler>
{
    public TransformCommand()
        : base("transform", "Transform a CSV data set file of features into a LETOR dataset")
    {
        var seedOption = new Option<int?>(["--seed", "-s"], "The seed to use for randomness. Many operations require randomness, so providing a fixed value makes operations deterministic. [default: a random value]");
        var inputOption = new Option<string>(["--input", "-i"], "The path to the input dataset. " +
            "The dataset is assumed to have a header of column names. If it does not, headers must be " +
            "provided with --headers.") { IsRequired = true };
        var outputOption = new Option<string?>(["--output", "-o"], "The path to the output LETOR dataset. " +
            "If unspecified, outputs to the current directory using the input filename suffixed with -letor");
        var headersOption = new Option<string?>(["--headers"], "The path to the input dataset headers file. " +
            "The headers file contains the headers for columns, separated with the same separator as the input file");
        var separatorOption = new Option<char?>(["--separator"], "The column separator character. [default: \\t]");
        var labelColumnOption = new Option<string>(["--label-column", "--label", "-l"], 
            () => nameof(DataPoint.Label), "The name of the label column. The label column value must be an unsigned integer value.");
        var queryIdColumnOption = new Option<string>(["--query-id-column", "--query", "-q"], 
            () => nameof(DataPoint.QueryId), "The name of the query ID column. The query ID column value must be a unsigned integer value.");
        var descriptionColumnOption = new Option<string?>(["--description-column", "--description", "-d"], 
            "The name of the description column");
        var featureColumnsOption = new Option<List<string>?>(["--feature-column", "--feature", "-f"], 
            "The name of the feature columns. If unspecified, all other columns are considered feature columns. Feature column values must be float values");

        AddOption(seedOption);
        AddOption(inputOption);
        AddOption(outputOption);
        AddOption(headersOption);
        AddOption(separatorOption);
        AddOption(labelColumnOption);
        AddOption(queryIdColumnOption);
        AddOption(descriptionColumnOption);
        AddOption(featureColumnsOption);
    }
}

internal class TransformCommandHandler : ICommandOptionsHandler<TransformCommandOptions>
{
    private readonly IAnsiConsole _ansiConsole;

    public TransformCommandHandler(IAnsiConsole ansiConsole) => _ansiConsole = ansiConsole;

    public Task<int> HandleAsync(TransformCommandOptions options, CancellationToken cancellationToken)
    {
        // set the seed here rather than in options default value, so that --help isn't confusing by
        // including a random value.
        options.Seed ??= RandomNumberGenerator.GetInt32(int.MaxValue);
        options.Separator ??= '\t';
        
        var optionsTable = new Table()
            .Title("[bold]Command Options[/]")
            .Border(TableBorder.Rounded)
            .AddColumn("Option")
            .AddColumn("Value");

        foreach (var kvp in options.ToDictionary()) 
            optionsTable.AddRow(kvp.Key, kvp.Value ?? "[grey]null[/]");
        _ansiConsole.Write(optionsTable);
        
        var mlContext = new MLContext(options.Seed);
        
        if (!TryGetColumnNames(options, out var columnNames))
            return Task.FromResult(1);

        if (!TryGetColumns(columnNames!, options, out var columns))
            return Task.FromResult(1);
        
        var loaderOptions = new TextLoader.Options
        {
            HasHeader = string.IsNullOrEmpty(options.Headers),
            TrimWhitespace = true,
            Separators = [options.Separator.Value],
            Columns = columns,
        };

        if (!string.IsNullOrEmpty(options.Headers)) 
            loaderOptions.HeaderFile = options.Headers;
        
        _ansiConsole.Status()
            .Spinner(Spinner.Known.Dots)
            .Start("Loading input data...", ctx =>
            {
                var inputData = mlContext.Data.LoadFromTextFile(options.Input, loaderOptions);
                
                ctx.Status("Transforming data...");
                
                var pipeline = CreatePipeline(mlContext, inputData, options, out var featureColumns);
                var model = pipeline.Fit(inputData);
                var transformedData = model.Transform(inputData);
                
                var output = !string.IsNullOrEmpty(options.Output) 
                    ? options.Output 
                    : $"{Path.GetFileNameWithoutExtension(options.Input)}-letor.txt";
                
                var featuresOutput = $"{Path.GetFileNameWithoutExtension(options.Input)}-features.txt";
                
                var directory = Path.GetDirectoryName(output);
                
                ctx.Status($"Saving data to {output}...");
                
                if (!string.IsNullOrEmpty(directory))
                    Directory.CreateDirectory(directory);
                
                Save(mlContext, transformedData, output);
                SaveFeatures(featureColumns, options.Input, featuresOutput);
            });
        
        return Task.FromResult(0);
    }

    private static void SaveFeatures(string[] featureColumns, string input, string path)
    {
        using var stream = File.Create(path);
        using var writer = new StreamWriter(stream);
        
        writer.Write("#\tFeatures from ");
        writer.WriteLine(input);
        for (var index = 0; index < featureColumns.Length; index++)
        {
            var featureColumn = featureColumns[index];
            writer.Write(index + 1);
            writer.Write('\t');
            writer.WriteLine(featureColumn);
        }

        writer.Flush();
    }

    private bool TryGetColumns(string[] columnNames, TransformCommandOptions options, out TextLoader.Column[]? columns)
    {
        columns = null;
        using var stream = File.OpenRead(options.Input);
        using var streamReader = new StreamReader(stream);
        var line = streamReader.ReadLine();
        if (string.IsNullOrWhiteSpace(line))
        {
            _ansiConsole.MarkupLine("[red]Input data is empty.[/]");
            return false;
        }
            
        // Assume that the file may have headers, so read the second line
        line = streamReader.ReadLine();
        if (string.IsNullOrWhiteSpace(line))
        {
            _ansiConsole.MarkupLine("[red]Input data is empty.[/]");
            return false;
        }
            
        var columnValues = line.Split(
            [options.Separator!.Value], 
            StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        columns = new TextLoader.Column[columnValues.Length];
        for (var i = 0; i < columnValues.Length; i++)
        {
            
            var columnName = columnNames[i];
            
            if (columnName == options.LabelColumn)
            {
                if (!uint.TryParse(columnValues[i], out _))
                {
                    _ansiConsole.MarkupLine($"[red]Column {columnName} cannot be parsed as a uint.[/]");
                    return false;
                }
                
                columns[i] = new TextLoader.Column(columnName, DataKind.UInt32, i);
                continue;
            }
            
            if (columnName == options.QueryIdColumn)
            {
                if (!uint.TryParse(columnValues[i], out _))
                {
                    _ansiConsole.MarkupLine($"[red]Column {columnName} cannot be parsed as a uint.[/]");
                    return false;
                }
                
                columns[i] = new TextLoader.Column(columnName, DataKind.UInt32, i);
                continue;
            }

            if (options.DescriptionColumn != null && columnName == options.DescriptionColumn)
            {
                columns[i] = new TextLoader.Column(columnName, DataKind.String, i);
                continue;
            }

            if (options.FeatureColumn is null)
            {
                if (!float.TryParse(columnValues[i], out _))
                {
                    _ansiConsole.MarkupLine($"[red]Column {columnName} cannot be parsed as a float.[/]");
                    return false;
                }
                
                columns[i] = new TextLoader.Column(columnName, DataKind.Single, i);
            }
            else if (options.FeatureColumn.Contains(columnName))
            {
                if (!float.TryParse(columnValues[i], out _))
                {
                    _ansiConsole.MarkupLine($"[red]Column {columnName} cannot be parsed as a float.[/]");
                    return false;
                }
                
                columns[i] = new TextLoader.Column(columnName, DataKind.Single, i);
            }
            else
            {
                columns[i] = new TextLoader.Column(columnName, DataKind.String, i);
            }
        }
        
        return true;
    }

    private bool TryGetColumnNames(TransformCommandOptions options, out string[]? columnNames)
    {
        var headersFile = options.Headers ?? options.Input;
        using var stream = File.OpenRead(headersFile);
        using var streamReader = new StreamReader(stream);
        var line = streamReader.ReadLine();
        if (string.IsNullOrWhiteSpace(line))
        {
            _ansiConsole.MarkupLine("[red]Could not get column names for the input dataset.[/]");
            columnNames = null;
            return false;
        }

        columnNames = line.Split(
            [options.Separator!.Value],
            StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        return true;
    }

    private static void Save(MLContext mlContext, IDataView data, string outputPath)
    {
        // ignore missing columns used in the event there is no Description column.
        // Prior validation ensures that other columns are present.
        var dataPoints = mlContext.Data.CreateEnumerable<DataPoint>(data, reuseRowObject: false, ignoreMissingColumns: true);
        LetorDataFileWriter.Write(outputPath, dataPoints);
    }

    private static  EstimatorChain<ColumnCopyingTransformer> CreatePipeline(
        MLContext mlContext, 
        IDataView data, 
        TransformCommandOptions options, 
        out string[] featureColumns)
    {
        if (options.FeatureColumn != null)
            featureColumns = options.FeatureColumn.ToArray();
        else
        {
            featureColumns = data.Schema.AsQueryable()
                .Select(s => s.Name)
                .Where(c =>
                    c != options.LabelColumn &&
                    c != options.QueryIdColumn &&
                    (string.IsNullOrEmpty(options.DescriptionColumn) || c != options.DescriptionColumn))
                .ToArray();
        }
        
        var dataPipeline = mlContext.Transforms
            .Concatenate(nameof(DataPoint.Features), featureColumns)
            .Append(mlContext.Transforms.CopyColumns(nameof(DataPoint.Label), inputColumnName: options.LabelColumn))
            .Append(mlContext.Transforms.CopyColumns(nameof(DataPoint.QueryId),
                inputColumnName: options.QueryIdColumn));

        if (!string.IsNullOrEmpty(options.DescriptionColumn))
            dataPipeline.Append(mlContext.Transforms.CopyColumns(nameof(DataPoint.Description),
                inputColumnName: options.DescriptionColumn));

        return dataPipeline;
    }
}