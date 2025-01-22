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
    /// Reads data points from a LETOR format file.
    /// </summary>
    /// <param name="path">The path to the file</param>
    /// <returns>An enumerable of <see cref="DataPoint"/></returns>
    public static IEnumerable<DataPoint> Read(string path)
    {
        var stream = File.OpenRead(path);
        return Read(stream);
    }
    
    /// <summary>
    /// Reads data points from a stream in LETOR format.
    /// </summary>
    /// <remarks>
    /// Closes the stream after use.
    /// </remarks>
    /// <param name="stream">The stream to read from</param>
    /// <returns>An enumerable of <see cref="DataPoint"/></returns>
    public static IEnumerable<DataPoint> Read(Stream stream)
    {
        using var reader = new StreamReader(stream);
        while (reader.ReadLine() is { } line)
        {
            var lineSpan = line.AsSpan();
            if (lineSpan.IsEmpty || lineSpan.StartsWith("#") || lineSpan.IsWhiteSpace()) 
                continue;
            
            yield return DataPoint.Parse(lineSpan);
        }
    }
}