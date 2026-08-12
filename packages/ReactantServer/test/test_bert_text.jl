# BertText convention tests. Pure Julia (no GPU): the shared tokenizer + wire padding that text
# bundles call from preprocess. The tokenizer itself is pinned against HuggingFace `tokenizers`
# encodings by the per-bundle parity suites (which run the real vocab.txt); these tests cover the
# invariants a bundle depends on: a tiny hand-written vocab exercises WordPiece/[UNK]/pair
# truncation, and the wire helpers are checked against the KServe layouts in the manifests.

const _BT = ReactantServer.BertText

# Minimal uncased vocab: specials first (ids 0..4, as in a real vocab.txt), then pieces.
const _BT_VOCAB = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "the", "cat", "sat", "un", "##able", "##s", "cafe", ".", "漢",
]

function _bt_tokenizer(f)
    return mktempdir() do dir
        path = joinpath(dir, "vocab.txt")
        write(path, join(_BT_VOCAB, "\n"))
        f(_BT.load_tokenizer(path))
    end
end

# (max_bytes, batch) UInt8 matrix + lens, the client wire layout for a batch of texts.
function _bt_wire(texts::Vector{String})
    rows = [Vector{UInt8}(codeunits(t)) for t in texts]
    mx = maximum(length, rows; init = 1)
    data = zeros(UInt8, mx, length(rows))
    for (b, r) in enumerate(rows)
        data[1:length(r), b] = r
    end
    return data, Int32[length(r) for r in rows]
end

@testset "BertText" begin
    @testset "vocab and special ids" begin
        _bt_tokenizer() do t
            @test t.pad_id == 0 && t.unk_id == 1 && t.cls_id == 2 && t.sep_id == 3
        end
        # a vocab missing a required special is a load-time error, not a serve-time surprise
        mktempdir() do dir
            path = joinpath(dir, "vocab.txt")
            write(path, join(["[PAD]", "[UNK]", "the"], "\n"))
            @test_throws ErrorException _BT.load_tokenizer(path)
        end
    end

    @testset "encode_single: template, WordPiece, [UNK], truncation" begin
        _bt_tokenizer() do t
            ids = _BT.encode_single(t, "the cat sat")
            @test ids == Int32[2, 5, 6, 7, 3]                  # [CLS] the cat sat [SEP]

            # greedy longest match with "##" continuations
            @test _BT.encode_single(t, "unables") == Int32[2, 8, 9, 10, 3]

            # a word with no matching piece collapses to a single [UNK]
            @test _BT.encode_single(t, "the zebra") == Int32[2, 5, 1, 3]

            # normalization: lowercase, strip accents, isolate punctuation, space CJK
            @test _BT.encode_single(t, "CAFÉ.") == Int32[2, 11, 12, 3]
            @test _BT.encode_single(t, "漢漢") == Int32[2, 13, 13, 3]

            # right truncation keeps room for both specials
            @test _BT.encode_single(t, "the cat sat"; max_len = 4) == Int32[2, 5, 6, 3]
            @test length(_BT.encode_single(t, repeat("the ", 50); max_len = 8)) == 8
        end
    end

    @testset "encode_pair: template, type ids, LongestFirst, empty b" begin
        _bt_tokenizer() do t
            ids, tt = _BT.encode_pair(t, "the cat", "sat")
            @test ids == Int32[2, 5, 6, 3, 7, 3]               # [CLS] a [SEP] b [SEP]
            @test tt == Int32[0, 0, 0, 0, 1, 1]                # the trailing [SEP] belongs to b

            # an empty b falls back to the single encoding with all-zero type ids
            ids, tt = _BT.encode_pair(t, "the cat", "")
            @test ids == _BT.encode_single(t, "the cat")
            @test all(iszero, tt)

            # LongestFirst: the longer side is cut to the shorter one first
            long, short = repeat("the ", 20), "cat sat"
            ids, tt = _BT.encode_pair(t, long, short; max_len = 12)
            @test length(ids) == 12 && length(tt) == 12
            @test count(==(2), ids) == 1 && count(==(3), ids) == 2
            @test sum(tt) == 3                                  # cat sat [SEP] on the b side
        end
    end

    @testset "seq_bucket" begin
        @test _BT.seq_bucket(1, (64, 128, 512)) == 64
        @test _BT.seq_bucket(64, (64, 128, 512)) == 64
        @test _BT.seq_bucket(65, (64, 128, 512)) == 128
        @test _BT.seq_bucket(9999, (64, 128, 512)) == 512       # clamps to the largest bucket
        @test _BT.seq_bucket(3, (512,)) == 512
    end

    @testset "wire helpers" begin
        _bt_tokenizer() do t
            @test _BT.wire_text(Vector{UInt8}(codeunits("the cat"))) == "the cat"
            @test _BT.wire_text(reshape(Vector{UInt8}(codeunits("cafe")), 4, 1)) == "cafe"

            data, lens = _bt_wire(["the cat", "sat", ""])
            @test _BT.wire_texts(data, lens) == ["the cat", "sat", ""]
            @test_throws ErrorException _BT.wire_texts(data, lens[1:2])
            @test_throws ErrorException _BT.wire_texts(data, Int32[99, 1, 1])
            @test_throws ErrorException _BT.wire_texts(data, Int32[-1, 1, 1])
        end
    end

    @testset "encode_text_batch: padded (seq, batch) executable inputs" begin
        _bt_tokenizer() do t
            data, lens = _bt_wire(["the cat sat", "the cat", ""])
            ids, mask = _BT.encode_text_batch(t, data, lens; max_len = 64, buckets = (64, 128))

            @test size(ids) == (64, 3) && size(mask) == (64, 3)
            @test eltype(ids) === Int64 && eltype(mask) === Int64
            @test ids[1, :] == fill(t.cls_id, 3)                # every row starts with [CLS]
            @test ids[1:5, 1] == Int64[2, 5, 6, 7, 3]
            @test ids[1:2, 3] == Int64[2, 3]                    # empty row is [CLS][SEP]
            @test vec(sum(mask; dims = 1)) == Int64[5, 4, 2]
            @test all(iszero, ids[6:end, 1])                    # 0 == [PAD]
            @test all(iszero, mask[6:end, 1])

            # the bucket follows the longest row in the batch
            short, slens = _bt_wire(["the"])
            sids, _ = _BT.encode_text_batch(t, short, slens; max_len = 64, buckets = (8, 64))
            @test size(sids) == (8, 1)

            # a max_len past the largest bucket fails with a clear message, not a BoundsError
            @test_throws ErrorException _BT.encode_text_batch(t, data, lens; max_len = 64, buckets = (4,))

            # error messages name the offending client tensor
            err = try
                _BT.encode_text_batch(t, data, lens[1:1]; lens_name = "text_lens")
            catch e
                e
            end
            @test occursin("text_lens", err.msg)
        end
    end

    @testset "encode_pair_batch: one query against N keys" begin
        _bt_tokenizer() do t
            data, lens = _bt_wire(["sat", "the cat sat", ""])
            ids, mask, tt = _BT.encode_pair_batch(
                t, "the cat", data, lens;
                max_len = 64, buckets = (64,)
            )

            @test size(ids) == (64, 3) && size(mask) == (64, 3) && size(tt) == (64, 3)
            @test ids[1:6, 1] == Int64[2, 5, 6, 3, 7, 3]
            @test count(==(t.sep_id), ids[:, 1]) == 2           # pair row has two [SEP]
            @test count(==(t.sep_id), ids[:, 3]) == 1           # empty key: single encoding
            @test tt[5:6, 1] == Int64[1, 1] && all(iszero, tt[:, 3])
            @test vec(sum(mask; dims = 1)) == Int64[6, 8, 4]
            @test all(iszero, ids[7:end, 1])
        end
    end
end
