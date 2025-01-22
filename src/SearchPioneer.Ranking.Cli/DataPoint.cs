namespace SearchPioneer.Ranking.Cli;

/// <summary>
/// Represents a single data point in the LETOR file format
/// </summary>
public record DataPoint : IDataPoint
{
    /// <summary>
    /// Instantiates a new instance of <see cref="DataPoint"/>
    /// </summary>
    /// <param name="label">The relevance label</param>
    /// <param name="queryId">The query ID. Used to group data points</param>
    /// <param name="features">A vector of features</param>
    /// <param name="description">A description for the data point. This typically includes the document ID and query input.</param>
    public DataPoint(uint label,
        uint queryId,
        float[] features,
        string description)
    {
        Label = label;
        QueryId = queryId;
        Features = features;
        Description = description;
    }

    /// <summary>
    /// Instantiates a new instance of <see cref="DataPoint"/>
    /// </summary>
    public DataPoint()
    {
    }
    
    /// <summary>
    /// The relevance label
    /// </summary>
    public uint Label { get; set; }
    
    /// <summary>
    /// The query ID. Used to group data points
    /// </summary>
    public uint QueryId { get; set; }
    
    /// <summary>
    /// A vector of features
    /// </summary>
    public float[] Features { get; set; } = [];
    
    /// <summary>
    /// A description for the data point. This typically includes the document ID and query input.
    /// </summary>
    public string Description { get; set; } = string.Empty;

    /// <summary>
    /// Attempt to parse a DataPoint from the provided span.
    /// </summary>
    /// <param name="span">The span</param>
    /// <param name="entry">The DataPoint if parsing is successful, null otherwise</param>
    /// <returns><c>true</c> if parsing is successful, <c>false</c> otherwise</returns>
    public static bool TryParse(ReadOnlySpan<char> span, out DataPoint? entry)
    {
        entry = null;
        span = span.Trim();
        if (span.IsEmpty)
            return false;
            
        var commentIndex = span.IndexOf('#');
        var comment = string.Empty;
        if (commentIndex != -1)
        {
            comment = span[commentIndex..].Trim().ToString();
            span = span[..commentIndex].Trim();
        }
        
        var parts = span.SplitOnWhitespace();

        if (!parts.MoveNext())
            return false;
        
        if (!uint.TryParse(parts.Current, out var label))
            return false;
        
        if (!parts.MoveNext())
            return false;

        if (!uint.TryParse(GetValue(parts.Current), out var queryId))
            return false;
        
        var features = new List<float>();
        while (parts.MoveNext())
        {
            if (!float.TryParse(GetValue(parts.Current), out var featureValue))
                return false;
            
            features.Add(featureValue);
        }

        if (features.Count == 0)
            return false;
        
        entry = new DataPoint(label, queryId, features.ToArray(), comment);
        return true;
    }
    
    /// <summary>
    /// Attempt to parse a DataPoint from the provided span.
    /// </summary>
    /// <param name="span">The span</param>
    /// <returns>A new instance of <see cref="DataPoint"/></returns>
    /// <exception cref="ArgumentException">
    /// Thrown if the span does not contain a valid DataPoint.
    /// </exception>
    public static DataPoint Parse(ReadOnlySpan<char> span)
    {
        if (!TryParse(span, out var entry))
            throw new ArgumentException($"Not a valid data point: '{span}'");
        
        return entry!;
    }
    
    /// <summary>
    /// Writes the DataPoint to the provided writer
    /// </summary>
    /// <param name="writer">The writer</param>
    public void WriteTo(TextWriter writer)
    {
        writer.Write(Label);
        writer.Write(' ');
        writer.Write("qid:");
        writer.Write(QueryId);
        writer.Write(' ');
        
        for (var i = 0; i < Features.Length; i++)
        {
            // LETOR dataset features are 1-based.
            writer.Write(i + 1);
            writer.Write(':');
            writer.Write(Features[i]);
            writer.Write(' ');
        }
        
        if (!string.IsNullOrEmpty(Description))
        {
            writer.Write("# ");
            writer.Write(Description);
        }
        
        writer.WriteLine();
    }
    
    private static ReadOnlySpan<char> GetValue(ReadOnlySpan<char> pair) => pair.Slice(pair.LastIndexOf(':') + 1);
}