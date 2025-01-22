using System.CommandLine;
using Bullseye;
using static BuildTargets;
using static Bullseye.Targets;
using static SimpleExec.Command;

var cmd = new RootCommand
{
	new Argument<string[]>("targets")
	{
		Description =
			"A list of targets to run or list. If not specified, the \"default\" target will be run, or all targets will be listed."
	}
};

foreach (var (aliases, description) in Options.Definitions)
	cmd.Add(new Option<bool>(aliases.ToArray(), description));

cmd.SetHandler(async () =>
{
	var cmdLine = cmd.Parse(args);
	var targets = cmdLine.CommandResult.Tokens.Select(token => token.Value);
	var options = new Options(Options.Definitions.Select(d => (d.Aliases[0],
		cmdLine.GetValueForOption(cmd.Options.OfType<Option<bool>>().Single(o => o.HasAlias(d.Aliases[0]))))));

	var configuration = "Release";
	var packOutput = "nuget";

	Target(Restore, () =>
	{
		Run("dotnet", "restore");
		Run("dotnet", "tool restore");
	});

	Target(Clean, () =>
	{
		Run("dotnet", $"clean -c {configuration} -v m --nologo");
	});

	Target(Format, DependsOn(Restore), () =>
	{
		Run("dotnet", "format");
	});

	Target(BuildSln, DependsOn(Restore, Clean), () =>
	{
		Run("dotnet", $"build -c {configuration} --nologo");
	});
	
	Target(Test, DependsOn(BuildSln), () =>
	{
		Run("dotnet", $"test -c {configuration} --no-build --verbosity normal");
	});
	
	Target(CleanPack, () =>
	{
		if (Directory.Exists(packOutput))
			Directory.Delete(packOutput, true);
	});

	Target(Pack, DependsOn(BuildSln, CleanPack), () =>
	{
		var outputDir = Directory.CreateDirectory(packOutput);
		Run("dotnet", $"pack -c {configuration} -o \"{outputDir.FullName}\" --no-build --nologo");
	});
	
	Target(Default, DependsOn(Test));

	await RunTargetsAndExitAsync(targets, options, messageOnly: ex => ex is SimpleExec.ExitCodeException);
});

return await cmd.InvokeAsync(args);

internal static class BuildTargets
{
	public const string Clean = "clean";
	public const string CleanPack = "clean-pack";
	public const string BuildSln = "build";
	public const string Test = "test";
	public const string Default = "default";
	public const string Restore = "restore";
	public const string Format = "format";
	public const string Pack = "pack";
}
