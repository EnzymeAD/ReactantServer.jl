# Self-contained BERT WordPiece tokenizer (uncased), matching HuggingFace `tokenizers`:
# BertNormalizer (clean_text, handle_chinese_chars, strip_accents, lowercase) +
# BertPreTokenizer (whitespace split, punctuation isolation) + WordPiece (greedy longest
# match, "##" continuation) + TemplateProcessing single/pair encodings with LongestFirst
# truncation. Special-token literals ([CLS], [SEP], ...) are matched verbatim in raw text
# before normalization, mirroring HF's AddedVocabulary. No dependencies beyond Base and
# the Unicode stdlib, so a bundle's model.jl can `include` it directly.
module BertWordPiece

using Unicode

export BertTokenizer, load_tokenizer, encode_single, encode_pair

struct BertTokenizer
    vocab::Dict{String,Int32}          # token -> 0-based id (KServe wire ids match HF)
    unk_id::Int32
    cls_id::Int32
    sep_id::Int32
    pad_id::Int32
    max_input_chars_per_word::Int
    specials::Vector{Pair{String,Int32}}   # matched verbatim in raw text, pre-normalization
end

function load_tokenizer(vocab_path::AbstractString; max_input_chars_per_word::Int=100)
    vocab = Dict{String,Int32}()
    for (i, line) in enumerate(eachline(vocab_path))
        vocab[line] = Int32(i - 1)
    end
    id(tok) = get(() -> error("vocab has no $tok"), vocab, tok)
    specials = [tok => id(tok) for tok in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]") if haskey(vocab, tok)]
    return BertTokenizer(vocab, id("[UNK]"), id("[CLS]"), id("[SEP]"), id("[PAD]"),
                         max_input_chars_per_word, specials)
end

# --- character classes (utf8proc category codes via Base.Unicode) ---

const _CAT_MN = 6                       # nonspacing mark
const _CAT_P = 12:18                    # Pc Pd Ps Pe Pi Pf Po
# Cc Cf Cs Co. Unassigned (Cn, code 0) is deliberately NOT control: HF's Rust
# unicode_categories crate has no table entry for unassigned codepoints, so BertNormalizer
# keeps them in the word (which then usually becomes [UNK]).
const _CAT_C = (26, 27, 28, 29)

# HF is_control: tab/newline/CR are whitespace, everything in Unicode group C is control.
function _is_control(c::Char)
    (c == '\t' || c == '\n' || c == '\r') && return false
    return Base.Unicode.category_code(c) in _CAT_C
end

# HF is_whitespace: Rust char::is_whitespace (White_Space property). Julia's isspace covers
# all of it except Zl (U+2028) and Zp (U+2029).
_is_ws(c::Char) = isspace(c) || c == '\u2028' || c == '\u2029'

# ASCII punctuation (includes $ + < = > ^ ` | ~, which are Unicode S*) or Unicode P*.
function _is_punct(c::Char)
    if isascii(c)
        u = UInt32(c)
        return (0x21 <= u <= 0x2f) || (0x3a <= u <= 0x40) || (0x5b <= u <= 0x60) || (0x7b <= u <= 0x7e)
    end
    return Base.Unicode.category_code(c) in _CAT_P
end

function _is_cjk(c::Char)
    u = UInt32(c)
    return (0x4E00 <= u <= 0x9FFF) || (0x3400 <= u <= 0x4DBF) ||
           (0x20000 <= u <= 0x2A6DF) || (0x2A700 <= u <= 0x2B73F) ||
           (0x2B740 <= u <= 0x2B81F) || (0x2B820 <= u <= 0x2CEAF) ||
           (0xF900 <= u <= 0xFAFF) || (0x2F800 <= u <= 0x2FA1F)
end

# --- BertNormalizer, in HF's order: clean_text, CJK spacing, strip accents, lowercase ---

function _normalize(s::AbstractString)
    io = IOBuffer(sizehint=ncodeunits(s) + 16)
    for c in s
        (c == '\0' || c == '�' || _is_control(c)) && continue
        if _is_cjk(c)
            write(io, ' '); write(io, c); write(io, ' ')
        else
            write(io, _is_ws(c) ? ' ' : c)
        end
    end
    nfd = Unicode.normalize(String(take!(io)), :NFD)
    io = IOBuffer(sizehint=ncodeunits(nfd))
    for c in nfd
        Base.Unicode.category_code(c) == _CAT_MN || write(io, c)
    end
    return lowercase(String(take!(io)))
end

# --- BertPreTokenizer: split on whitespace, isolate each punctuation char ---

function _pretokenize(s::AbstractString)
    words = String[]
    buf = IOBuffer()
    flush!() = (buf.size > 0 && push!(words, String(take!(buf))); nothing)
    for c in s
        if _is_ws(c)
            flush!()
        elseif _is_punct(c)
            flush!()
            push!(words, string(c))
        else
            write(buf, c)
        end
    end
    flush!()
    return words
end

# --- WordPiece: greedy longest-match; any unmatched piece collapses the word to [UNK] ---

function _wordpiece!(out::Vector{Int32}, t::BertTokenizer, word::AbstractString)
    chars = collect(word)
    n = length(chars)
    if n > t.max_input_chars_per_word
        push!(out, t.unk_id)
        return
    end
    pieces = Int32[]
    i = 1
    while i <= n
        j = n
        found = Int32(-1)
        while j >= i
            cand = i > 1 ? "##" * String(chars[i:j]) : String(chars[i:j])
            id = get(t.vocab, cand, Int32(-1))
            if id >= 0
                found = id
                break
            end
            j -= 1
        end
        if found < 0
            push!(out, t.unk_id)
            return
        end
        push!(pieces, found)
        i = j + 1
    end
    append!(out, pieces)
    return
end

# Tokenize raw text to content ids (no [CLS]/[SEP]). Special-token literals in the raw
# text map straight to their ids, as HF's added-token matcher does before normalization.
function _tokenize(t::BertTokenizer, text::AbstractString)
    out = Int32[]
    pos = firstindex(text)
    stop = lastindex(text)
    while pos <= stop
        best_at = typemax(Int)
        best = nothing
        for (tok, id) in t.specials
            r = findnext(tok, text, pos)
            r === nothing && continue
            if first(r) < best_at
                best_at = first(r)
                best = (r, id)
            end
        end
        seg_end = best === nothing ? stop : prevind(text, best_at)
        if pos <= seg_end
            for w in _pretokenize(_normalize(SubString(text, pos, seg_end)))
                _wordpiece!(out, t, w)
            end
        end
        best === nothing && break
        push!(out, best[2])
        pos = nextind(text, last(best[1]))
    end
    return out
end

# --- encodings (TemplateProcessing + LongestFirst right truncation) ---

"""
    encode_single(t, text; max_len=512) -> Vector{Int32}

`[CLS] text [SEP]`, truncated on the right to `max_len` total. token_type_ids are all zero.
"""
function encode_single(t::BertTokenizer, text::AbstractString; max_len::Int=512)
    ids = _tokenize(t, text)
    keep = max_len - 2
    length(ids) > keep && resize!(ids, keep)
    return Int32[t.cls_id; ids; t.sep_id]
end

"""
    encode_pair(t, a, b; max_len=512) -> (ids, type_ids)

`[CLS] a [SEP] b [SEP]` with HF LongestFirst truncation to `max_len` total: the longer
sequence is cut down to the shorter one first, then the remaining excess is split evenly,
with the odd token removed from the originally shorter side (ties treat `b` as longer).
An empty `b` string falls back to the single encoding, matching transformers' falsy
`text_pair` handling.
"""
function encode_pair(t::BertTokenizer, a::AbstractString, b::AbstractString; max_len::Int=512)
    if isempty(b)
        ids = encode_single(t, a; max_len=max_len)
        return ids, zeros(Int32, length(ids))
    end
    ta = _tokenize(t, a)
    tb = _tokenize(t, b)
    la, lb = length(ta), length(tb)
    excess = la + lb - (max_len - 3)
    if excess > 0
        if la > lb
            d = min(excess, la - lb); la -= d; excess -= d
            la -= excess ÷ 2; lb -= cld(excess, 2)
        else
            d = min(excess, lb - la); lb -= d; excess -= d
            lb -= excess ÷ 2; la -= cld(excess, 2)
        end
        resize!(ta, max(la, 0))
        resize!(tb, max(lb, 0))
    end
    ids = Int32[t.cls_id; ta; t.sep_id; tb; t.sep_id]
    type_ids = Int32[zeros(Int32, length(ta) + 2); ones(Int32, length(tb) + 1)]
    return ids, type_ids
end

end # module
