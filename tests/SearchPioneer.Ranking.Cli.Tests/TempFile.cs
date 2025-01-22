namespace SearchPioneer.Ranking.Cli.Tests;

public class TempFile : IDisposable
{
    private readonly FileInfo _fileInfo;

    public TempFile()
    {
        var tempFileName = System.IO.Path.GetTempFileName();
        _fileInfo = new FileInfo(tempFileName);
    }

    public string Path => _fileInfo.FullName;

    public StreamWriter GetWriter() => new(_fileInfo.FullName);

    public void Dispose()
    {
        try
        {
            _fileInfo.Refresh();
            if (_fileInfo.Exists)
                _fileInfo.Delete();
        }
        catch (Exception)
        {
            // suppress
        }
    }
}