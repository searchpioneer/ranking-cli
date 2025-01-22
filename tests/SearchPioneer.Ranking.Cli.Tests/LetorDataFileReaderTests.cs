using System.Text;
using SearchPioneer.Ranking.Cli;
using Xunit;

namespace SearchPioneer.Ranking.Cli.Tests;

public class LetorDataFileReaderTests
{
    private const string Dataset = @"
4 qid:1 1:12.318474 2:10.573917 # 7555 rambo
3 qid:1 1:10.357876 2:11.95039  # 1370 rambo
3 qid:1 1:7.0105133 2:11.220095 # 1369 rambo
3 qid:1 1:0.0       2:11.220095 # 1368 rambo
0 qid:1 1:0.0       2:0.0       # 136278 rambo
0 qid:1 1:0.0       2:0.0       # 102947 rambo
0 qid:1 1:0.0       2:0.0       # 13969 rambo
0 qid:1 1:0.0       2:0.0       # 61645 rambo
0 qid:1 1:0.0       2:0.0       # 14423 rambo
0 qid:1 1:0.0       2:0.0       # 54156 rambo
4 qid:2 1:10.686391 2:8.814846  # 1366 rocky
3 qid:2 1:8.985554  2:9.984511  # 1246 rocky
3 qid:2 1:8.985554  2:8.067703  # 60375 rocky
3 qid:2 1:8.985554  2:5.66055   # 1371 rocky
3 qid:2 1:8.985554  2:7.300773  # 1375 rocky
3 qid:2 1:8.985554  2:8.814846  # 1374 rocky
0 qid:2 1:6.815921  2:0.0       # 110123 rocky
0 qid:2 1:6.081685  2:8.725065  # 17711 rocky
0 qid:2 1:6.081685  2:5.9764786 # 36685 rocky
4 qid:3 1:7.672084  2:12.72242  # 17711 bullwinkle
0 qid:3 1:0.0       2:0.0       # 1246 bullwinkle
0 qid:3 1:0.0       2:0.0       # 60375 bullwinkle
0 qid:3 1:0.0       2:0.0       # 1371 bullwinkle
0 qid:3 1:0.0       2:0.0       # 1375 bullwinkle
0 qid:3 1:0.0       2:0.0       # 1374 bullwinkle";
    
    [Fact]
    public void Can_read_valid_input_from_stream()
    {
        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(Dataset));
        var items = LetorDataFileReader.Read(stream).ToList();
        Assert.Equal(25, items.Count);
    }
    
    [Fact]
    public void Can_read_valid_input_from_file()
    {
        var file = new TempFile();
        using (var writer = file.GetWriter())
        {
            writer.Write(Dataset);
        }

        var items = LetorDataFileReader.Read(file.Path).ToList();
        Assert.Equal(25, items.Count);
    }
}