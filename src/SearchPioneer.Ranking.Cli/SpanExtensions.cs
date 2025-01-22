namespace SearchPioneer.Ranking.Cli;

internal static class SpanExtensions
{
    public static WhitespaceSplitEnumerator SplitOnWhitespace(this ReadOnlySpan<char> span) => new(span);
}