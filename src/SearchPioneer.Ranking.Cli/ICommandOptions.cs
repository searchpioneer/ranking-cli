using System.CommandLine;
using System.CommandLine.NamingConventionBinder;
using Microsoft.Extensions.DependencyInjection;

namespace SearchPioneer.Ranking.Cli;

// From:
// https://anthonysimmon.com/true-dependency-injection-with-system-commandline/
// Licensed under CC BY 4.0

/// <summary>
/// Marker interface for command options
/// </summary>
public interface ICommandOptions
{
    /// <summary>
    /// Dictionary of command names and values to output to the console.
    /// </summary>
    Dictionary<string, string?> ToDictionary();
}

/// <summary>
/// A handler for a command
/// </summary>
/// <typeparam name="TOptions">The type of options for the command</typeparam>
public interface ICommandOptionsHandler<in TOptions>
{
    /// <summary>
    /// Handler for the command options
    /// </summary>
    /// <param name="options">The options</param>
    /// <param name="cancellationToken">A cancellation token</param>
    /// <returns>A task with the exit code</returns>
    Task<int> HandleAsync(TOptions options, CancellationToken cancellationToken);
}

/// <summary>
/// Base class for a command with options and a handler.
/// </summary>
/// <typeparam name="TOptions">The type of options for the command</typeparam>
/// <typeparam name="TOptionsHandler">The type of handler for the command</typeparam>
public abstract class Command<TOptions, TOptionsHandler> : Command
    where TOptions : class, ICommandOptions
    where TOptionsHandler : class, ICommandOptionsHandler<TOptions>
{
    protected Command(string name, string description)
        : base(name, description) =>
        Handler = CommandHandler.Create<TOptions, IServiceProvider, CancellationToken>(HandleOptions);

    private static async Task<int> HandleOptions(TOptions options, IServiceProvider serviceProvider, CancellationToken cancellationToken)
    {
        var handler = ActivatorUtilities.CreateInstance<TOptionsHandler>(serviceProvider);
        try
        {
            return await handler.HandleAsync(options, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception e)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            await Console.Error.WriteLineAsync(e.Message);
            Console.ResetColor();
            return 1;
        }
    }
}