namespace SearchPioneer.Ranking.Cli;

internal static class LetorDataFileWriter
{
    public static void Write(string path, IEnumerable<DataPoint> dataPoints)
    {
        using var stream = File.Create(path);
        using var writer = new StreamWriter(stream);
        foreach (var dataPoint in dataPoints) 
            dataPoint.WriteTo(writer);
            
        writer.Flush();
    }
}