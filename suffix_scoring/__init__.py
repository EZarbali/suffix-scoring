"""High-throughput suffix scoring with Qwen3-0.6B.

score(suffix | prefix) = log P(t1|prefix) + log P(t2|prefix,t1) + log P(t3|prefix,t1,t2)
for 400 three-token candidates [a-z][A-Z][0-9] per prefix.
"""
__version__ = "1.0.0"
