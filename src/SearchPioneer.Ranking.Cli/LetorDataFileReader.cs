namespace SearchPioneer.Ranking.Cli;

/// <summary>
/// Reader for LETOR format files
/// </summary>
/// <remarks>
/// See
/// <a href="https://arxiv.org/pdf/1306.2597">
/// Introducing LETOR 4.0 Datasets
/// </a> and
/// <a href="https://searchpioneer.github.io/ranklib-dotnet/">
/// RankLib API reference
/// </a>
/// </remarks>
internal static class LetorDataFileReader
{
    /// <summary>
    /// Reads data points from a file in LETOR format.
    /// </summary>
    /// <param name="path">The input path to read from</param>
    /// <returns>An enumerable of <see cref="DataPoint"/></returns>
    public static IEnumerable<DataPoint> Read(string path)
    {
        using var stream = File.OpenRead(path);
        using var reader = new StreamReader(stream, leaveOpen: true);
        while (reader.ReadLine() is { } line)
        {
            var lineSpan = line.AsSpan();
            if (lineSpan.IsEmpty || lineSpan.StartsWith("#") || lineSpan.IsWhiteSpace())
                continue;

            yield return DataPoint.Parse(lineSpan);
        }
    }
}