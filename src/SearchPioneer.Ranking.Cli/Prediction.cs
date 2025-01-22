namespace SearchPioneer.Ranking.Cli;

/// <summary>
/// The prediction from the model
/// </summary>
public class Prediction : IDataPoint
{
    /// <inheritdoc />
    public uint QueryId { get; set; }
    
    /// <inheritdoc />
    public uint Label { get; set; }
    
    /// <summary>
    /// Prediction made by the model that is used to indicate the relative ranking of the candidate search results
    /// </summary>
    public float Score { get; set; }

    /// <inheritdoc />
    public float[] Features { get; set; } = [];
    
    /// <inheritdoc />
    public string Description { get; set; } = default!;
}