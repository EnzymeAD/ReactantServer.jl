# Julia text glue for BERT-family text bundles, the request-side counterpart to DetectionGlue:
# a self-contained BERT WordPiece tokenizer plus the wire-format helpers a text bundle's
# preprocess needs. Referenced from a bundle's model.jl as ReactantServer.BertText. Depends on
# nothing but Base (Base.Unicode), so it can also be included standalone by an export-side or
# offline test script.
#
# The tokenizer matches HuggingFace `tokenizers`: BertNormalizer (clean_text,
# handle_chinese_chars, strip_accents, lowercase) + BertPreTokenizer (whitespace split,
# punctuation isolation) + WordPiece (greedy longest match, "##" continuation) +
# TemplateProcessing single/pair encodings with LongestFirst truncation. Special-token literals
# ([CLS], [SEP], ...) are matched verbatim in raw text before normalization, mirroring HF's
# AddedVocabulary. The vocabulary itself stays per-bundle (vocab.txt next to model.jl): it is
# checkpoint-specific, only the code is shared.
#
# Wire conventions (KServe row-major on the wire, so col-major here): a batch of texts arrives as
# a UInt8 matrix (max_bytes, batch) of zero-padded UTF-8 rows plus an INT32 length per row; the
# executable inputs come back as (seq, batch) Int64 matrices with 0 == [PAD].

module BertText

# --- tokenizer ---

struct BertTokenizer
    vocab::Dict{String,Int32}          # token -> 0-based id (KServe wire ids match HF)
    unk_id::Int32
    cls_id::Int32
    sep_id::Int32
    pad_id::Int32
    max_input_chars_per_word::Int
    specials::Vector{Pair{String,Int32}}   # matched verbatim in raw text, pre-normalization
end

"""
    load_tokenizer(vocab_path; max_input_chars_per_word=100) -> BertTokenizer

Load a HuggingFace `vocab.txt` (one token per line, line number - 1 is the id).
"""
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
    nfd = Base.Unicode.normalize(String(take!(io)), :NFD)
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

# --- wire helpers: UInt8 client tensors -> padded executable inputs ---

"""
    seq_bucket(n, buckets) -> Int

Smallest compiled sequence length in `buckets` (ascending) that holds `n` tokens; the largest
bucket if none does. Padding a request up to its bucket is bit-identical to a tight fit because
every op downstream is attention-mask aware.
"""
function seq_bucket(n::Integer, buckets)
    i = findfirst(>=(n), buckets)
    return Int(i === nothing ? last(buckets) : buckets[i])
end

"""
    wire_text(bytes) -> String

Decode one UTF-8 client tensor (any UInt8 array shape) to a String. Use for a scalar text input
such as a cross-encoder's shared query; batched rows go through [`wire_texts`](@ref).
"""
wire_text(bytes::AbstractArray{UInt8}) = String(collect(UInt8, vec(bytes)))

"""
    wire_texts(texts, lens; lens_name="lens") -> Vector{String}

Decode a `(max_bytes, batch)` UInt8 matrix of zero-padded UTF-8 rows into `batch` Strings, using
`lens[b]` as row `b`'s byte length. `lens_name` only names the offending tensor in error messages.
"""
function wire_texts(texts::AbstractMatrix{UInt8}, lens::AbstractVector;
                    lens_name::AbstractString="lens")
    B = size(texts, 2)
    length(lens) == B || error("$lens_name has $(length(lens)) entries for $B text rows")
    out = Vector{String}(undef, B)
    for b in 1:B
        n = Int(lens[b])
        0 <= n <= size(texts, 1) || error("$lens_name[$b] = $n out of range")
        out[b] = String(@view texts[1:n, b])
    end
    return out
end

# Bucket the batch and check the encodings fit, so an over-long max_len fails with a clear
# message instead of a BoundsError while filling the padded matrices.
function _bucket_for(encoded::AbstractVector{<:AbstractVector}, buckets)
    maxlen = 1
    for ids in encoded
        maxlen = max(maxlen, length(ids))
    end
    seq = seq_bucket(maxlen, buckets)
    maxlen <= seq ||
        error("encoded $maxlen tokens but the largest sequence bucket is $seq; lower max_len " *
              "to the largest bucket or compile a longer one")
    return seq
end

"""
    pad_batch(encoded, buckets) -> (input_ids, attention_mask)

Pack per-row token ids into `(seq, batch)` Int64 matrices at the smallest bucket that fits,
zero-padded (0 == [PAD]) with a 1/0 attention mask.
"""
function pad_batch(encoded::AbstractVector{<:AbstractVector{Int32}}, buckets)
    B = length(encoded)
    seq = _bucket_for(encoded, buckets)
    input_ids = zeros(Int64, seq, B)
    attention_mask = zeros(Int64, seq, B)
    for b in 1:B, (i, id) in enumerate(encoded[b])
        input_ids[i, b] = id
        attention_mask[i, b] = 1
    end
    return input_ids, attention_mask
end

"""
    pad_batch(encoded_pairs, buckets) -> (input_ids, attention_mask, token_type_ids)

Pair-encoding form of [`pad_batch`](@ref): each entry is an `(ids, type_ids)` tuple.
"""
function pad_batch(encoded::AbstractVector{<:Tuple{AbstractVector{Int32},AbstractVector{Int32}}},
                   buckets)
    B = length(encoded)
    seq = _bucket_for([e[1] for e in encoded], buckets)
    input_ids = zeros(Int64, seq, B)
    attention_mask = zeros(Int64, seq, B)
    token_type_ids = zeros(Int64, seq, B)
    for b in 1:B
        ids, type_ids = encoded[b]
        for i in eachindex(ids)
            input_ids[i, b] = ids[i]
            attention_mask[i, b] = 1
            token_type_ids[i, b] = type_ids[i]
        end
    end
    return input_ids, attention_mask, token_type_ids
end

"""
    encode_text_batch(t, texts, lens; max_len=512, buckets=(512,), lens_name="lens")
        -> (input_ids, attention_mask)

One-text-per-row preprocess: decode the `(max_bytes, batch)` UInt8 rows, encode each as
`[CLS] text [SEP]`, and pad to the smallest bucket that fits. Both outputs are `(seq, batch)`
Int64 matrices ready to hand back as executable inputs.
"""
function encode_text_batch(t::BertTokenizer, texts::AbstractMatrix{UInt8}, lens::AbstractVector;
                           max_len::Int=512, buckets=(512,), lens_name::AbstractString="lens")
    rows = wire_texts(texts, lens; lens_name=lens_name)
    encoded = [encode_single(t, s; max_len=max_len) for s in rows]
    return pad_batch(encoded, buckets)
end

"""
    encode_pair_batch(t, query, keys, lens; max_len=512, buckets=(512,), lens_name="lens")
        -> (input_ids, attention_mask, token_type_ids)

Cross-encoder preprocess: score one `query` against every row of the `(max_bytes, batch)` UInt8
`keys` matrix, encoding each as `[CLS] query [SEP] key [SEP]` with LongestFirst truncation. An
empty key row falls back to the single encoding (all-zero token_type_ids), matching HF.
"""
function encode_pair_batch(t::BertTokenizer, query::AbstractString, keys::AbstractMatrix{UInt8},
                           lens::AbstractVector; max_len::Int=512, buckets=(512,),
                           lens_name::AbstractString="lens")
    rows = wire_texts(keys, lens; lens_name=lens_name)
    encoded = [encode_pair(t, query, key; max_len=max_len) for key in rows]
    return pad_batch(encoded, buckets)
end

end # module
