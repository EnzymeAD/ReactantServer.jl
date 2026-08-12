# ReactantServer.jl conventions

## CONTRIBUTING.md

**Always read it first**

## Formatting: Runic.jl

**Format Julia source with [Runic.jl](https://github.com/fredrikekre/Runic.jl) before committing.**
Run `runic --inplace packages docs examples tools` (all tracked source; `lib/` holds only the
gRPCServer.jl submodule and is not formatted here). The pre-commit hook runs this automatically.

## Prose

**No em dashes or en dashes in new or edited text**, in code, comments, docstrings, documents, or
commit messages. Use a comma, a colon, a semicolon, or restructure the sentence. This is **not
retroactive**: leave existing prose alone, and only apply it to content you write or touch.

Commit messages carry the reasoning, not just the change.
