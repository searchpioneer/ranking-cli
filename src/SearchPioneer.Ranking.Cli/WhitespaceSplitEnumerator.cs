namespace SearchPioneer.Ranking.Cli;

internal ref struct WhitespaceSplitEnumerator
{
    private ReadOnlySpan<char> _remaining;
    private ReadOnlySpan<char> _current;

    public WhitespaceSplitEnumerator(ReadOnlySpan<char> span)
    {
        _remaining = span;
        _current = default;
    }

    public bool MoveNext()
    {
        while (_remaining.Length > 0 && char.IsWhiteSpace(_remaining[0]))
            _remaining = _remaining[1..];

        if (_remaining.Length == 0)
            return false;

        var end = 0;
        while (end < _remaining.Length && !char.IsWhiteSpace(_remaining[end]))
            end++;

        _current = _remaining[..end];
        _remaining = _remaining[end..];
        return true;
    }

    public ReadOnlySpan<char> Current => _current;

    public WhitespaceSplitEnumerator GetEnumerator() => this;
}