namespace SearchPioneer.Ranking.Cli;

/// <summary>
/// Represents a single data point
/// </summary>
public interface IDataPoint
{
    /// <summary>
    /// The relevance label
    /// </summary>
    uint Label { get; set; }

    /// <summary>
    /// The query ID. Used to group data points
    /// </summary>
    uint QueryId { get; set; }

    /// <summary>
    /// A vector of features
    /// </summary>
    float[] Features { get; set; }

    /// <summary>
    /// A description for the data point. This typically includes the document ID and query input.
    /// </summary>
    string Description { get; set; }
}