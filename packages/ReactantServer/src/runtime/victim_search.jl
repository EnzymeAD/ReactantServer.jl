# Victim-set search for weight admission: which resident models to evict to free a byte deficit
# at the lowest total eviction cost (a min-cost covering knapsack).
#
# It runs on the dispatch thread before every on-demand load that needs room, so it is built to be
# fast and to allocate nothing: the candidate arrays are owned by the weight cache and reused, and
# the search is anytime with a fixed budget. In order:
#
#   1. Greedy by cost per byte, then drop any taken model the cover does not need.
#   2. The cheapest single model that covers the deficit alone.
#      The better of 1 and 2 costs at most twice the optimum, so the result is bounded even if
#      step 3 is cut off at once.
#   3. Branch and bound in cost-per-byte order, pruned by the fractional (LP) relaxation, which
#      the same order makes a greedy fill. It stops at `SEARCH_NODE_BUDGET` nodes or when the
#      bound proves the incumbent optimal, and returns the best cover found.
#
# Candidates are filled by the caller (admission.jl). `protected` marks models evicted only when
# the others cannot cover the deficit (queued requests, minimum residency).

# Branch-and-bound nodes per search. A node is one O(N) bound evaluation, so with tens of
# candidates this caps a search at well under a millisecond; realistic cases prove optimality far
# sooner.
const SEARCH_NODE_BUDGET = 1_000

mutable struct VictimSearch
    names::Vector{String}       # candidate model names (references, so filling allocates nothing)
    bytes::Vector{Int}          # bytes evicting each candidate frees
    cost::Vector{Float64}       # cost of evicting each candidate (>= 0)
    protected::Vector{Bool}
    order::Vector{Int}          # eligible candidate indices, ascending cost per byte
    take::Vector{Bool}          # branch-and-bound assignment, by position in `order`
    best::Vector{Bool}          # the chosen cover, by candidate index
    n::Int                      # candidates filled
    m::Int                      # eligible candidates in `order`
    best_cost::Float64
    nodes::Int
end

VictimSearch() = VictimSearch(String[], Int[], Float64[], Bool[], Int[], Bool[], Bool[], 0, 0, Inf, 0)

"""
    reset_candidates!(s, n) -> s

Size the candidate arrays for `n` candidates. Allocates only when `n` exceeds every previous size.
"""
function reset_candidates!(s::VictimSearch, n::Int)
    # One call per vector: a loop over a tuple of differently typed vectors would dispatch dynamically.
    resize!(s.names, n); resize!(s.bytes, n); resize!(s.cost, n); resize!(s.protected, n)
    resize!(s.order, n); resize!(s.take, n); resize!(s.best, n)
    s.n = n
    s.m = 0
    return s
end

@inline function set_candidate!(s::VictimSearch, i::Int, name::String, bytes::Int, cost::Float64, protected::Bool)
    @inbounds begin
        s.names[i] = name
        s.bytes[i] = bytes
        s.cost[i] = cost
        s.protected[i] = protected
    end
    return nothing
end

@inline _density(s::VictimSearch, i::Int) = @inbounds s.cost[i] / max(s.bytes[i], 1)

# Fill `order` with the eligible candidates sorted by cost per byte, ties by candidate index (the
# caller fills in LRU order, so ties evict the least recently used first). Insertion sort: N is
# tens, and it allocates nothing.
function _order_eligible!(s::VictimSearch, allow_protected::Bool)
    m = 0
    @inbounds for i in 1:s.n
        (allow_protected || !s.protected[i]) || continue
        m += 1
        s.order[m] = i
        j = m
        while j > 1
            a = s.order[j - 1]
            (_density(s, a) > _density(s, i) || (_density(s, a) == _density(s, i) && a > i)) || break
            s.order[j] = a
            j -= 1
        end
        s.order[j] = i
    end
    s.m = m
    return m
end

# Lower bound on the cost of covering `rem` more bytes from positions `k..m` of `order`: fill
# fractionally in density order. Inf when the remaining candidates cannot cover it.
function _fractional_bound(s::VictimSearch, k::Int, rem::Int)
    acc = 0.0
    @inbounds for p in k:s.m
        i = s.order[p]
        b = s.bytes[i]
        b >= rem && return acc + s.cost[i] * (rem / b)
        acc += s.cost[i]
        rem -= b
    end
    return Inf
end

function _record_best!(s::VictimSearch, k::Int, cost::Float64)
    s.best_cost = cost
    fill!(s.best, false)
    @inbounds for p in 1:(k - 1)
        s.take[p] && (s.best[s.order[p]] = true)
    end
    return nothing
end

function _branch!(s::VictimSearch, k::Int, covered::Int, cost::Float64, deficit::Int)
    s.nodes += 1
    if covered >= deficit
        cost < s.best_cost && _record_best!(s, k, cost)
        return nothing
    end
    (k > s.m || s.nodes > SEARCH_NODE_BUDGET) && return nothing
    _fractional_bound(s, k, deficit - covered) + cost >= s.best_cost && return nothing
    i = @inbounds s.order[k]
    @inbounds s.take[k] = true
    _branch!(s, k + 1, covered + (@inbounds s.bytes[i]), cost + (@inbounds s.cost[i]), deficit)
    @inbounds s.take[k] = false
    _branch!(s, k + 1, covered, cost, deficit)
    return nothing
end

"""
    select_cover!(s, deficit; allow_protected=false) -> Bool

Choose a set of candidates freeing at least `deficit` bytes at low total cost, into `s.best`
(by candidate index). Returns false, choosing nothing, when the eligible candidates together cannot
cover the deficit. Allocates nothing.
"""
function select_cover!(s::VictimSearch, deficit::Integer; allow_protected::Bool = false)
    deficit = Int(deficit)
    fill!(s.best, false)
    s.best_cost = Inf
    s.nodes = 0
    m = _order_eligible!(s, allow_protected)
    total = 0
    @inbounds for p in 1:m
        total += s.bytes[s.order[p]]
    end
    total >= deficit || return false

    # 1. Greedy by density, then drop what the cover does not need (most expensive per byte first).
    covered = 0
    last = 0
    @inbounds for p in 1:m
        covered >= deficit && break
        s.take[p] = true
        covered += s.bytes[s.order[p]]
        last = p
    end
    @inbounds for p in last:-1:1
        i = s.order[p]
        if covered - s.bytes[i] >= deficit
            s.take[p] = false
            covered -= s.bytes[i]
        end
    end
    greedy = 0.0
    @inbounds for p in 1:last
        s.take[p] && (greedy += s.cost[s.order[p]])
    end
    _record_best!(s, last + 1, greedy)

    # 2. The cheapest single candidate that covers the deficit alone.
    @inbounds for p in 1:m
        i = s.order[p]
        if s.bytes[i] >= deficit && s.cost[i] < s.best_cost
            s.best_cost = s.cost[i]
            fill!(s.best, false)
            s.best[i] = true
        end
    end

    # 3. Bounded improvement; skipped when the relaxation already proves the incumbent optimal.
    if _fractional_bound(s, 1, deficit) < s.best_cost
        fill!(s.take, false)
        _branch!(s, 1, 0, 0.0, deficit)
    end
    return true
end

"""
    select_cheapest!(s; allow_protected=false) -> Int

The single cheapest candidate per byte (ties: lowest index, i.e. least recently used), or 0 when
none is eligible. Used when contiguity rather than byte count is short, where no byte deficit
defines a cover. Allocates nothing.
"""
function select_cheapest!(s::VictimSearch; allow_protected::Bool = false)
    best = 0
    @inbounds for i in 1:s.n
        (allow_protected || !s.protected[i]) || continue
        (best == 0 || _density(s, i) < _density(s, best)) && (best = i)
    end
    return best
end
