using System.CommandLine;
using System.CommandLine.Builder;
using System.CommandLine.Parsing;
using Microsoft.Extensions.DependencyInjection;
using SearchPioneer.Ranking.Cli;
using Spectre.Console;

var rootCommand = new RootCommand("Train and evaluate ranking models using LightGBM and FastTree")
{
    new TrainCommand(),
    new SplitCommand(),
    new FoldCommand(),
    new TransformCommand(),
};

// Use the ToolCommandName when installed as a .NET tool
rootCommand.Name = "dotnet-ranking";

var builder = new CommandLineBuilder(rootCommand)
    .UseDefaults()
    .UseDependencyInjection(services =>
    {
        services.AddSingleton(AnsiConsole.Console);
    });

return await builder.Build().InvokeAsync(args);