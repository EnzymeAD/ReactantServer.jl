# Generates the Grafana dashboards in `dashboards/` from the definitions below. Edit this file, not
# the JSON. Needs only Julia's standard library.
#
#   julia config/monitoring/grafana/build.jl                          # regenerate dashboards/*.json
#   julia config/monitoring/grafana/build.jl --check http://host:9090 # run every query against a Prometheus
#
# `--check` substitutes the dashboard variables (every worker and model, a one-hour range) and
# reports each query that fails to parse or evaluate, and each that returns nothing. An empty result
# is often expected (no failures, no evictions, a single worker), so only errors fail the check.
#
# Conventions, kept the same on every dashboard:
#   * the datasource is the `ds` variable, so one set of dashboards serves any deployment;
#   * every dashboard opens with a strip of headline numbers, then rows that each answer one question;
#   * a "top N" panel ranks over the whole time range (`@ end()`), so its series do not change from one
#     step to the next, and `$topn` sets N;
#   * "over the range" numbers use `[$__range]`, so they cover exactly the window on screen;
#   * events (loads, evictions, repacks) are drawn per minute;
#   * clicking a model anywhere opens Model Detail for it.
#
# PromQL here uses single-quoted strings so the JSON needs no escaping, and short tokens for the
# dashboard variables, expanded by `p`: %W = the worker filter, %M = the model filter, %m = the one
# model on Model Detail, %ri = [$__rate_interval], %rg = [$__range], %n = $topn, %job = the job filter.

using Downloads

const OUT = joinpath(@__DIR__, "dashboards")
const TAG = "reactantserver"
const DS = Dict("type" => "prometheus", "uid" => "\${ds}")

const TOKENS = Dict(
    "%W" => "worker=~'\$worker'", "%M" => "model=~'\$model'", "%m" => "model='\$model'",
    "%ri" => "[\$__rate_interval]", "%rg" => "[\$__range]", "%n" => "\$topn", "%job" => "job=~'\$job'",
)
p(s::AbstractString) = replace(s, r"%(ri|rg|job|W|M|m|n)" => t -> TOKENS[t])

# ---- JSON ---------------------------------------------------------------------------------------

function json(io::IO, x, ind::Int = 0)
    pad(n) = "  "^n
    if x === nothing
        print(io, "null")
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Integer
        print(io, x)
    elseif x isa AbstractFloat
        isfinite(x) || error("non-finite number in a dashboard")
        print(io, isinteger(x) && abs(x) < 1.0e15 ? string(Int(x)) : repr(x))
    elseif x isa Union{AbstractString, Symbol}
        print(io, '"')
        for c in string(x)
            c == '"' ? print(io, "\\\"") : c == '\\' ? print(io, "\\\\") : c == '\n' ? print(io, "\\n") :
                c == '\t' ? print(io, "\\t") : c < ' ' ? print(io, "\\u", string(UInt16(c); base = 16, pad = 4)) :
                print(io, c)
        end
        print(io, '"')
    elseif x isa AbstractDict
        isempty(x) && return print(io, "{}")
        println(io, "{")
        ks = sort!(collect(keys(x)); by = string)
        for (i, k) in enumerate(ks)
            print(io, pad(ind + 1)); json(io, string(k)); print(io, ": "); json(io, x[k], ind + 1)
            println(io, i < length(ks) ? "," : "")
        end
        print(io, pad(ind), "}")
    elseif x isa Union{AbstractVector, Tuple}
        isempty(x) && return print(io, "[]")
        println(io, "[")
        for (i, v) in enumerate(x)
            print(io, pad(ind + 1)); json(io, v, ind + 1)
            println(io, i < length(x) ? "," : "")
        end
        print(io, pad(ind), "]")
    else
        error("cannot encode $(typeof(x)) as JSON")
    end
    return nothing
end

# ---- Panels -------------------------------------------------------------------------------------

D(pairs...) = Dict{String, Any}(string(k) => v for (k, v) in pairs)

"""A query: `expr` (with tokens), its legend, and whether it is an instant (table/stat) query."""
q(expr; legend = "", instant = false, fmt = nothing) =
    D("expr" => p(expr), "legendFormat" => legend, "instant" => instant, "range" => !instant, "format" => something(fmt, "time_series"))

steps(base, pairs...) = D("mode" => "absolute", "steps" => vcat([D("color" => base, "value" => nothing)], [D("color" => c, "value" => v) for (v, c) in pairs]))

const MODEL_LINK = "/d/rs-model/rs-model?var-ds=\${ds}&var-model=\${__field.labels.model}&\${__url_time_range}"
const MODEL_CELL_LINK = "/d/rs-model/rs-model?var-ds=\${ds}&var-model=\${__value.raw}&\${__url_time_range}"
model_links() = [D("title" => "Model detail: \${__field.labels.model}", "url" => MODEL_LINK)]

# Overrides, by series name pattern.
by_regex(re, props...) = D("matcher" => D("id" => "byRegexp", "options" => re), "properties" => [D("id" => k, "value" => v) for (k, v) in props])
by_name(name, props...) = D("matcher" => D("id" => "byName", "options" => name), "properties" => [D("id" => k, "value" => v) for (k, v) in props])
const DASHED = D("dash" => [6, 4], "fill" => "dash")
dashed(re) = by_regex(re, "custom.lineStyle" => DASHED, "custom.fillOpacity" => 0, "custom.stacking" => D("mode" => "none", "group" => "A"))

function panel(
        type, title, targets; desc = "", unit = nothing, decimals = nothing, min = nothing, max = nothing,
        thresholds = steps("green"), mappings = [], overrides = [], links = nothing, custom = D(),
        options = D(), color = D("mode" => "palette-classic"), w = 12, h = 8, extra = D()
    )
    targets = targets isa Dict ? [targets] : targets
    for (i, t) in enumerate(targets)
        t["refId"] = string(Char('A' + i - 1))
        t["datasource"] = DS
    end
    defaults = D("thresholds" => thresholds, "mappings" => mappings, "color" => color, "custom" => custom)
    unit === nothing || (defaults["unit"] = unit)
    decimals === nothing || (defaults["decimals"] = decimals)
    min === nothing || (defaults["min"] = min)
    max === nothing || (defaults["max"] = max)
    links === nothing || (defaults["links"] = links)
    d = D(
        "type" => type, "title" => title, "description" => desc, "datasource" => DS, "targets" => targets,
        "fieldConfig" => D("defaults" => defaults, "overrides" => overrides), "options" => options,
        "gridPos" => D("w" => w, "h" => h, "x" => 0, "y" => 0),
    )
    return merge!(d, extra)
end

function ts(
        title, targets; stack = false, bars = false, legend = :table, calcs = ["mean", "max", "lastNotNull"],
        fill = 10, custom = D(), kw...
    )
    c = D(
        "drawStyle" => bars ? "bars" : "line", "lineWidth" => 1, "fillOpacity" => bars ? 80 : (stack ? 35 : fill),
        "gradientMode" => "opacity", "showPoints" => "never", "spanNulls" => false, "axisSoftMin" => 0,
        "stacking" => D("mode" => stack ? "normal" : "none", "group" => "A"), "thresholdsStyle" => D("mode" => "off"),
        "lineInterpolation" => "linear", "barAlignment" => 0,
    )
    merge!(c, custom)
    lg = legend === :table ?
        D("showLegend" => true, "displayMode" => "table", "placement" => "right", "calcs" => calcs, "sortBy" => "Mean", "sortDesc" => true) :
        legend === :hidden ? D("showLegend" => false, "displayMode" => "list", "placement" => "bottom", "calcs" => []) :
        D("showLegend" => true, "displayMode" => "list", "placement" => "bottom", "calcs" => [])
    opts = D("legend" => lg, "tooltip" => D("mode" => "multi", "sort" => "desc"))
    return panel("timeseries", title, targets; custom = c, options = opts, kw...)
end

function stat(title, targets; graph = false, calc = "lastNotNull", text = "auto", kw...)
    opts = D(
        "reduceOptions" => D("calcs" => [calc], "fields" => "", "values" => false),
        "colorMode" => "value", "graphMode" => graph ? "area" : "none", "textMode" => text,
        "justifyMode" => "auto", "orientation" => "auto", "showPercentChange" => false, "wideLayout" => true,
    )
    return panel("stat", title, targets; options = opts, color = D("mode" => "thresholds"), w = 3, h = 4, kw...)
end

function bars(title, targets; kw...)
    opts = D(
        "reduceOptions" => D("calcs" => ["lastNotNull"], "fields" => "", "values" => false),
        "orientation" => "horizontal", "displayMode" => "gradient", "showUnfilled" => true,
        "valueMode" => "color", "namePlacement" => "auto", "sizing" => "auto", "minVizHeight" => 16, "maxVizHeight" => 300,
    )
    return panel("bargauge", title, targets; options = opts, color = D("mode" => "continuous-GrYlRd"), kw...)
end

function pie(title, targets; kw...)
    opts = D(
        "reduceOptions" => D("calcs" => ["lastNotNull"], "fields" => "", "values" => false), "pieType" => "donut",
        "legend" => D("showLegend" => true, "displayMode" => "table", "placement" => "right", "values" => ["percent"]),
        "tooltip" => D("mode" => "single", "sort" => "none"), "displayLabels" => [],
    )
    return panel("piechart", title, targets; options = opts, kw...)
end

function heat(title, expr; unit = "s", kw...)
    t = q(expr; legend = "{{le}}", fmt = "heatmap")
    opts = D(
        "calculate" => false, "cellGap" => 1, "yAxis" => D("unit" => unit, "axisPlacement" => "left"),
        "color" => D("mode" => "scheme", "scheme" => "Oranges", "fill" => "dark-orange", "scale" => "exponential", "exponent" => 0.5, "steps" => 64, "reverse" => false),
        "rowsFrame" => D("layout" => "auto"), "tooltip" => D("mode" => "single", "yHistogram" => true, "showColorScale" => false),
        "legend" => D("show" => true), "exemplars" => D("color" => "rgba(255,0,255,0.7)"), "filterValues" => D("le" => 1.0e-9),
        "cellValues" => D("unit" => "short"),
    )
    return panel("heatmap", title, [t]; options = opts, kw...)
end

const UPDOWN = [D("type" => "value", "options" => D("0" => D("text" => "down", "color" => "red", "index" => 0), "1" => D("text" => "up", "color" => "green", "index" => 1)))]

function timeline(title, targets; mappings = UPDOWN, kw...)
    opts = D(
        "mergeValues" => true, "showValue" => "never", "alignValue" => "left", "rowHeight" => 0.9,
        "legend" => D("showLegend" => false, "displayMode" => "list", "placement" => "bottom"),
        "tooltip" => D("mode" => "single", "sort" => "none"),
    )
    return panel(
        "state-timeline", title, targets; mappings = mappings, options = opts, color = D("mode" => "thresholds"),
        thresholds = steps("red", (1, "green")), custom = D("fillOpacity" => 80, "lineWidth" => 0), kw...
    )
end

text(title, content; w = 24, h = 6) = D(
    "type" => "text", "title" => title, "options" => D("mode" => "markdown", "content" => content),
    "gridPos" => D("w" => w, "h" => h, "x" => 0, "y" => 0),
)

"""A table column: one instant query per column, joined on `by` by the merge transformation."""
struct Col
    title::String
    expr::String
    unit::Union{String, Nothing}
    decimals::Union{Int, Nothing}
end
Col(title, expr; unit = nothing, decimals = nothing) = Col(title, expr, unit, decimals)

function table(title, by::String, cols::Vector{Col}; sort = nothing, desc = "", model_link = by == "model", w = 24, h = 12, extra_overrides = [])
    targets = [q(c.expr; instant = true, fmt = "table") for c in cols]
    rename = Dict{String, Any}()
    index = Dict{String, Any}(by => 0)
    for (i, c) in enumerate(cols)
        rename["Value #$(Char('A' + i - 1))"] = c.title
        index[c.title] = i
    end
    overrides = Any[]
    for c in cols
        props = Any[]
        c.unit === nothing || push!(props, "unit" => c.unit)
        c.decimals === nothing || push!(props, "decimals" => c.decimals)
        isempty(props) || push!(overrides, by_name(c.title, props...))
    end
    model_link && push!(
        overrides, by_name(
            by, "links" => [D("title" => "Model detail: \${__value.raw}", "url" => MODEL_CELL_LINK)]
        )
    )
    append!(overrides, extra_overrides)
    opts = D("showHeader" => true, "cellHeight" => "sm", "footer" => D("show" => false, "reducer" => ["sum"], "fields" => ""))
    sort === nothing || (opts["sortBy"] = [D("displayName" => sort, "desc" => true)])
    excl = D("Time" => true)
    d = panel(
        "table", title, targets; desc = desc, options = opts, overrides = overrides, w = w, h = h,
        custom = D("align" => "auto", "filterable" => true, "cellOptions" => D("type" => "auto"), "inspect" => false),
        extra = D(
            "transformations" => [
                D("id" => "merge", "options" => D()),
                D("id" => "organize", "options" => D("excludeByName" => excl, "renameByName" => rename, "indexByName" => index)),
            ]
        ),
    )
    return d
end

# ---- Layout -------------------------------------------------------------------------------------

mutable struct Board
    uid::String
    title::String
    desc::String
    vars::Vector{Any}
    panels::Vector{Any}
    y::Int
    id::Int
    open_row::Any        # a collapsed row collecting its panels, or nothing
end
Board(uid, title, desc, vars) = Board(uid, title, desc, vars, Any[], 0, 0, nothing)

"""Start a row. A collapsed row holds the panels added after it until the next row."""
function row!(b::Board, title; collapsed = false)
    b.id += 1
    r = D("type" => "row", "title" => title, "id" => b.id, "collapsed" => collapsed, "panels" => Any[], "gridPos" => D("h" => 1, "w" => 24, "x" => 0, "y" => b.y))
    push!(b.panels, r)
    b.y += 1
    b.open_row = collapsed ? r : nothing
    return b
end

"""Place panels left to right on one line; their widths should add up to 24."""
function line!(b::Board, ps...)
    x = 0
    h = maximum(pp["gridPos"]["h"] for pp in ps)
    for pp in ps
        b.id += 1
        pp["id"] = b.id
        pp["gridPos"]["x"] = x
        pp["gridPos"]["y"] = b.y
        x += pp["gridPos"]["w"]
        push!(b.open_row === nothing ? b.panels : b.open_row["panels"], pp)
    end
    x <= 24 || error("$(b.uid): a line is $x wide")
    b.y += h
    return b
end

# Variables.
dsvar() = D("name" => "ds", "label" => "Prometheus", "type" => "datasource", "query" => "prometheus", "current" => D(), "hide" => 0, "refresh" => 1, "regex" => "", "options" => [])
function qvar(name, label, query; multi = true, all = true, hide = 0)
    return D(
        "name" => name, "label" => label, "type" => "query", "datasource" => DS, "query" => query, "definition" => query,
        "refresh" => 2, "multi" => multi, "includeAll" => all, "sort" => 1, "hide" => hide, "options" => [], "regex" => "",
        "current" => all ? D("text" => ["All"], "value" => ["\$__all"]) : D(),
    )
end
function custom(name, label, values, default)
    opts = [D("text" => v, "value" => v, "selected" => v == default) for v in values]
    return D(
        "name" => name, "label" => label, "type" => "custom", "query" => join(values, ","), "options" => opts,
        "current" => D("text" => default, "value" => default), "hide" => 0, "multi" => false, "includeAll" => false,
    )
end
jobvar() = qvar("job", "Job", "label_values(gateway_routing_table_size, job)"; hide = 2)
workervar() = qvar("worker", "Worker", "label_values(worker_info, worker)")
modelvar() = qvar("model", "Model", "label_values(worker_model_max_batch_size{worker=~'\$worker'}, model)")
topnvar() = custom("topn", "Top N", ["5", "10", "15", "20", "30"], "10")

# Annotations shared by every dashboard: repacks move models between workers (expect loads after
# one), and a worker restart resets its counters. Each marks one point per event: the value now minus
# the value one step earlier, evaluated every step, telescopes, so every change lands in exactly one
# step. A windowed `changes(x[w]) > 0` is true at every step while events come faster than `w`, and
# Grafana draws consecutive true steps as one shaded region instead of a line per event. Beyond about
# four days on screen Grafana widens the step past 30 s and some events go unmarked.
const ANNOTATIONS = [
    D(
        "builtIn" => 1, "datasource" => D("type" => "grafana", "uid" => "-- Grafana --"), "enable" => true, "hide" => true,
        "iconColor" => "rgba(0, 211, 255, 1)", "name" => "Annotations & Alerts", "type" => "dashboard",
    ),
    D(
        "name" => "Repacks", "datasource" => DS, "enable" => true, "iconColor" => "purple", "step" => "30s",
        "expr" => "sum(gateway_repacks_total) - sum(gateway_repacks_total offset 30s) > 0", "titleFormat" => "Repack",
        "textFormat" => "the gateway revised the placement (it may have moved nothing)",
    ),
    D(
        "name" => "Worker restarts", "datasource" => DS, "enable" => true, "iconColor" => "red", "step" => "30s",
        "expr" => "max by (worker) (process_start_time_seconds{worker!=''}) - max by (worker) (process_start_time_seconds{worker!=''} offset 30s) > 0",
        "titleFormat" => "{{worker}} restarted", "textFormat" => "",
    ),
]

function render(b::Board)
    return D(
        "uid" => b.uid, "title" => b.title, "description" => b.desc, "tags" => [TAG], "timezone" => "browser",
        "schemaVersion" => 39, "version" => 1, "editable" => true, "graphTooltip" => 1, "refresh" => "30s",
        "time" => D("from" => "now-6h", "to" => "now"), "timepicker" => D(), "fiscalYearStartMonth" => 0, "liveNow" => false,
        "links" => [
            D(
                "title" => "ReactantServer", "type" => "dashboards", "tags" => [TAG], "asDropdown" => false, "includeVars" => false,
                "keepTime" => true, "icon" => "external link", "targetBlank" => false, "tooltip" => "", "url" => "",
            ),
        ],
        "annotations" => D("list" => ANNOTATIONS), "templating" => D("list" => b.vars), "panels" => b.panels,
    )
end

# ---- Shared expressions ---------------------------------------------------------------------------

# Device busy: device-seconds of execution per second of wall time, per worker. Timed on the host
# around each execution, so it is "the device had work", not SM occupancy.
const BUSY = "sum by (worker) (rate(worker_compute_seconds_total{%W}%ri))"
# Rank models over the range by a counter, and keep the top N.
top(inner, counter, filt = "%W,%M") = "($inner) and on (model) topk(%n, sum by (model) (increase($counter{$filt}%rg @ end())))"
per_min(x) = "($x) * 60"

# ---- Overview -------------------------------------------------------------------------------------

function overview()
    b = Board(
        "rs-overview", "ReactantServer: Overview",
        "Is the node healthy and keeping up? Headline health, traffic, latency and saturation, with links to the detail dashboards.",
        [dsvar(), jobvar(), workervar(), topnvar()]
    )
    line!(
        b,
        stat(
            "Workers not ready", q("count(gateway_worker_ready == 0) or vector(0)");
            thresholds = steps("green", (1, "red")), desc = "Workers whose last ServerReady probe failed or timed out."
        ),
        stat(
            "Metrics not answering", q("count(gateway_worker_metrics_up == 0) or vector(0)");
            thresholds = steps("green", (1, "orange")), desc = "Workers whose metrics endpoint missed the last scrape; their panels go stale."
        ),
        stat("Request rate", q("sum(rate(gateway_requests_total{rpc='ModelInfer'}%ri))"); unit = "reqps", graph = true, decimals = 2),
        stat(
            "Success over the range",
            q("sum(increase(gateway_requests_total{rpc='ModelInfer',status='OK'}%rg)) / clamp_min(sum(increase(gateway_requests_total{rpc='ModelInfer'}%rg)), 1)");
            unit = "percentunit", decimals = 2, thresholds = steps("red", (0.99, "orange"), (0.999, "green"))
        ),
        stat(
            "p95 latency over the range",
            q("histogram_quantile(0.95, sum by (le) (increase(gateway_request_latency_seconds_bucket{rpc='ModelInfer'}%rg)))");
            unit = "s", thresholds = steps("green", (1, "orange"), (5, "red")), desc = "End to end through the gateway, queueing included."
        ),
        stat(
            "Shed over the range",
            q("(sum(increase(gateway_requests_shed_total%rg)) or vector(0)) + (sum(increase(worker_requests_shed_total%rg)) or vector(0))");
            thresholds = steps("green", (1, "red")), decimals = 0, desc = "Requests refused at a concurrency cap, at the gateway or a worker."
        ),
        stat(
            "Devices busy", q("avg($BUSY)"); unit = "percentunit", graph = true, decimals = 1,
            thresholds = steps("green", (0.7, "orange"), (0.9, "red")),
            desc = "Mean over the selected workers of the share of time each device spent executing models."
        ),
        stat(
            "Health probe p99, worst worker",
            q("max(histogram_quantile(0.99, sum by (worker, le) (increase(gateway_worker_probe_seconds_bucket{%W}%rg))))");
            unit = "s", thresholds = steps("green", (1, "orange"), (5, "red")),
            desc = "The ServerReady round trip. A worker whose request handling stalls (a blocked dispatch thread) shows here before anything else; 5 s is the probe's deadline."
        ),
    )
    line!(
        b,
        stat("Models", q("max(gateway_routing_table_size)"); desc = "Models the gateway can route."),
        stat("Resident on a device", q("sum(worker_models_resident{%W})"); desc = "Model copies whose weights are on a device now, summed over workers."),
        stat("Weight loads over the range", q("sum(increase(worker_weight_loads_total{%W}%rg))"); decimals = 0),
        stat(
            "Read from weights files", q("sum(increase(worker_weight_read_bytes_total{%W}%rg))"); unit = "bytes",
            desc = "Loads that found no host copy and read the file. On a node whose host weight budget is below the total weights, this is the expensive part of churn."
        ),
        stat(
            "Fragmentation, worst worker", q("max(worker_device_memory_fragmentation{%W})"); unit = "percentunit", decimals = 0,
            thresholds = steps("green", (0.3, "orange"), (0.6, "red")),
            desc = "1 minus the largest free block over the free bytes. Memory and Weights shows each worker's arena."
        ),
        stat(
            "Evictions for contiguity", q("sum(increase(worker_weight_evicts_total{%W,reason='contiguity'}%rg))"); decimals = 0,
            thresholds = steps("green", (1, "orange")), desc = "Over the range: evictions made although enough bytes were free, because no block held the new weights."
        ),
        stat("Repacks over the range", q("sum(increase(gateway_repacks_total%rg))"); decimals = 0),
        stat("Weights moved by repacks", q("sum(increase(gateway_repack_moved_bytes_total%rg))"); unit = "bytes"),
    )

    row!(b, "Traffic and latency")
    line!(
        b,
        ts("Requests by status", q("sum by (status) (rate(gateway_requests_total{rpc='ModelInfer'}%ri))"; legend = "{{status}}"); unit = "reqps", stack = true),
        ts(
            "Gateway latency",
            [
                q("histogram_quantile(0.5, sum by (le) (rate(gateway_request_latency_seconds_bucket{rpc='ModelInfer'}%ri)))"; legend = "p50"),
                q("histogram_quantile(0.95, sum by (le) (rate(gateway_request_latency_seconds_bucket{rpc='ModelInfer'}%ri)))"; legend = "p95"),
                q("histogram_quantile(0.99, sum by (le) (rate(gateway_request_latency_seconds_bucket{rpc='ModelInfer'}%ri)))"; legend = "p99"),
            ]; unit = "s"
        ),
    )
    line!(
        b,
        ts(
            "Failed requests by model and status", q("sum by (model, status) (rate(gateway_requests_total{rpc='ModelInfer',status!='OK'}%ri)) > 0"; legend = "{{model}} {{status}}");
            unit = "reqps", links = model_links()
        ),
        ts(
            "Requests, top N models", q(top("sum by (model) (rate(gateway_requests_total{rpc='ModelInfer'}%ri))", "gateway_requests_total", "rpc='ModelInfer'"); legend = "{{model}}");
            unit = "reqps", stack = true, links = model_links()
        ),
    )

    row!(b, "Saturation")
    line!(
        b,
        ts(
            "Device busy by worker", [q(BUSY; legend = "{{worker}}"), q("avg($BUSY)"; legend = "mean")]; unit = "percentunit", max = 1,
            overrides = [dashed("mean")], w = 8
        ),
        ts(
            "In-flight work by worker", q("sum by (worker, basis) (gateway_worker_inflight_work{%W})"; legend = "{{worker}}");
            w = 8, desc = "The per-worker load the gateway routes on, in the units of scheduling.work_basis (the basis label)."
        ),
        ts("Queued requests by worker", q("sum by (worker) (worker_queue_depth{%W})"; legend = "{{worker}}"); w = 8),
    )
    line!(
        b,
        ts(
            "Health probe round trip, p99 by worker",
            q("histogram_quantile(0.99, sum by (worker, le) (rate(gateway_worker_probe_seconds_bucket{%W}%ri)))"; legend = "{{worker}}");
            unit = "s", w = 12, thresholds = steps("green", (5, "red")), custom = D("thresholdsStyle" => D("mode" => "dashed")),
            desc = "Above a second means the worker answered late; at the 5 s line the gateway treats it as unavailable."
        ),
        ts(
            "Weight loads and evictions per minute", [
                q(per_min("sum(rate(worker_weight_loads_total{%W}%ri))"); legend = "loaded"),
                q(per_min("sum by (reason) (rate(worker_weight_evicts_total{%W}%ri))"); legend = "evicted: {{reason}}"),
            ]; w = 12, desc = "Churn across the selected workers. Memory and Weights breaks it down."
        ),
    )

    row!(b, "Readiness")
    line!(
        b,
        timeline("Worker ready", q("max by (worker) (gateway_worker_ready{%W})"; legend = "{{worker}}"); w = 12, h = 6),
        timeline("Worker metrics answering", q("max by (worker) (gateway_worker_metrics_up{%W})"; legend = "{{worker}}"); w = 12, h = 6),
    )
    return b
end

# ---- Models and Batching ----------------------------------------------------------------------------

const BATCHING_HELP = """
**Rows per dispatch** is the effective batch size: the batch-axis rows each device execution processed, whether the server merged separate requests or a client sent them pre-batched. **Requests per dispatch** counts only the server's merging, so a client that pre-batches reads 1 there while the device runs full batches.

**Fill** is rows per dispatch over the model's largest compiled batch. Low fill on a busy model means the device is running small batches: more concurrent callers, or client-side batching, would raise throughput.

Request counts and latency cover models called through the gateway. A meta model's sub-models are called inside the worker, so they appear in the dispatch, compute and batching panels only.
"""

function models()
    b = Board(
        "rs-models", "ReactantServer: Models and Batching",
        "Which models use the devices, how fast they answer, and how well they batch.",
        [dsvar(), workervar(), modelvar(), topnvar()]
    )
    line!(
        b,
        stat("Models loaded", q("max(sum by (worker) (worker_models_loaded{%W}))"); desc = "On the worker with the most."),
        stat("Copies resident", q("sum(worker_models_resident{%W})")),
        stat("Inference rate", q("sum(rate(worker_requests_total{%W,%M}%ri))"); unit = "reqps", graph = true, decimals = 2),
        stat("Inferences over the range", q("sum(increase(worker_requests_total{%W,%M}%rg))"); decimals = 0),
        stat("Failed over the range", q("sum(increase(worker_requests_total{%W,%M,status!='OK'}%rg)) or vector(0)"); decimals = 0, thresholds = steps("green", (1, "red"))),
        stat("Dispatch rate", q("sum(rate(worker_dispatch_total{%W,%M}%ri))"); unit = "ops", graph = true, decimals = 2),
        stat("Devices busy", q("avg($BUSY)"); unit = "percentunit", decimals = 1, thresholds = steps("green", (0.7, "orange"), (0.9, "red"))),
        stat(
            "Mean effective batch", q("sum(increase(worker_rows_served_total{%W,%M}%rg)) / clamp_min(sum(increase(worker_dispatch_total{%W,%M}%rg)), 1)");
            decimals = 2, desc = "Rows per dispatch over the range, across the selected models."
        ),
    )
    disp = "sum by (model) (increase(worker_dispatch_total{%W,%M}%rg))"
    devs = "sum by (model) (increase(worker_compute_seconds_total{%W,%M}%rg))"
    line!(
        b,
        table(
            "Models over the range", "model", [
                Col("Calls", "sum by (model) (increase(worker_requests_total{%W,%M}%rg))"; decimals = 0),
                Col("Failed", "sum by (model) (increase(worker_requests_total{%W,%M,status!='OK'}%rg))"; decimals = 0),
                Col("Dispatches", disp; decimals = 0),
                Col("Device time", devs; unit = "s"),
                Col("Device share", "$devs / scalar(clamp_min(sum(increase(worker_compute_seconds_total{%W}%rg)), 1e-9))"; unit = "percentunit", decimals = 1),
                Col("Per dispatch", "$devs / ($disp > 0)"; unit = "s"),
                Col("Rows/dispatch", "sum by (model) (increase(worker_rows_served_total{%W,%M}%rg)) / ($disp > 0)"; decimals = 2),
                Col("Max batch", "max by (model) (worker_model_max_batch_size{%W,%M})"; decimals = 0),
                Col("p95 latency", "histogram_quantile(0.95, sum by (model, le) (increase(worker_request_latency_seconds_bucket{%W,%M}%rg)))"; unit = "s"),
                Col("Queue wait p99", "max by (model) (worker_queue_wait_seconds{%W,%M,quantile='0.99'})"; unit = "s"),
                Col("Loads", "sum by (model) (increase(worker_model_loads_total{%W,%M}%rg))"; decimals = 0),
                Col("Weights", "max by (model) (worker_model_weight_bytes{%W,%M})"; unit = "bytes"),
                Col("Resident on", "sum by (model) (worker_model_resident{%W,%M})"; decimals = 0),
            ]; sort = "Device time", h = 14,
            desc = "Every model on the selected workers. Calls and latency are handler-side (gateway-fronted models); dispatches and device time include meta sub-models. Queue wait p99 is the worker's current window, not the range. Click a model for its detail."
        ),
    )

    row!(b, "Demand and cost")
    line!(
        b,
        ts("Inference rate, top N", q(top("sum by (model) (rate(worker_requests_total{%W,%M}%ri))", "worker_requests_total"); legend = "{{model}}"); unit = "reqps", stack = true, links = model_links()),
        ts(
            "Device time per second, top N", q(top("sum by (model) (rate(worker_compute_seconds_total{%W,%M}%ri))", "worker_compute_seconds_total"); legend = "{{model}}");
            unit = "percentunit", stack = true, links = model_links(), desc = "Stacked, this is how busy the devices are and with what (1 = one device fully busy)."
        ),
    )
    line!(
        b,
        pie(
            "Share of device time over the range", q("topk(%n, $devs)"; legend = "{{model}}", instant = true); unit = "s", links = model_links(),
            desc = "The top N models; the rest are left out, so shares are of the top N."
        ),
        bars("Calls over the range, top N", q("topk(%n, sum by (model) (increase(worker_requests_total{%W,%M}%rg)))"; legend = "{{model}}", instant = true); decimals = 0, links = model_links()),
    )

    row!(b, "Latency and queueing")
    line!(
        b,
        ts(
            "p95 handler latency, top N by calls",
            q(top("histogram_quantile(0.95, sum by (model, le) (rate(worker_request_latency_seconds_bucket{%W,%M}%ri)))", "worker_requests_total"); legend = "{{model}}");
            unit = "s", links = model_links()
        ),
        heat("Handler latency distribution, selected models", "sum by (le) (rate(worker_request_latency_seconds_bucket{%W,%M}%ri))"),
    )
    line!(
        b,
        ts("Queue wait p99, top N", q(top("max by (model) (worker_queue_wait_seconds{%W,%M,quantile='0.99'})", "worker_dispatch_total"); legend = "{{model}}"); unit = "s", links = model_links()),
        ts("Queued requests by model", q("sum by (model) (worker_queue_depth{%W,%M}) > 0"; legend = "{{model}}"); stack = true, links = model_links()),
    )
    line!(
        b,
        ts(
            "Device time per dispatch, top N",
            q(top("sum by (model) (rate(worker_compute_seconds_total{%W,%M}%ri)) / (sum by (model) (rate(worker_dispatch_total{%W,%M}%ri)) > 0)", "worker_compute_seconds_total"); legend = "{{model}}");
            unit = "s", links = model_links()
        ),
        ts("Requests in flight by worker", q("sum by (worker) (worker_inflight_requests{%W})"; legend = "{{worker}}"); desc = "Counted only when the worker's concurrency cap is on."),
    )

    row!(b, "Batching")
    line!(
        b,
        text("How to read batching", BATCHING_HELP; w = 8, h = 10),
        ts(
            "Effective batch size (rows per dispatch), top N",
            q(top("sum by (model) (rate(worker_rows_served_total{%W,%M}%ri)) / (sum by (model) (rate(worker_dispatch_total{%W,%M}%ri)) > 0)", "worker_dispatch_total"); legend = "{{model}}");
            w = 16, h = 10, links = model_links()
        ),
    )
    line!(
        b,
        bars(
            "Batch fill over the range, top N by dispatches",
            q(
                "(sum by (model) (increase(worker_rows_served_total{%W,%M}%rg)) / (sum by (model) (increase(worker_dispatch_total{%W,%M}%rg)) > 0) / (max by (model) (worker_model_max_batch_size{%W,%M}) > 0)) and on (model) topk(%n, sum by (model) (increase(worker_dispatch_total{%W,%M}%rg)))";
                legend = "{{model}}", instant = true
            ); unit = "percentunit", min = 0, max = 1, links = model_links(), h = 10,
            color = D("mode" => "continuous-RdYlGr"), desc = "Green is fuller batches, red is emptier."
        ),
        ts(
            "Requests merged per dispatch, top N",
            q(top("sum by (model) (rate(worker_requests_served_total{%W,%M}%ri)) / (sum by (model) (rate(worker_dispatch_total{%W,%M}%ri)) > 0)", "worker_dispatch_total"); legend = "{{model}}");
            h = 10, links = model_links()
        ),
    )

    row!(b, "Failures"; collapsed = true)
    line!(
        b,
        ts(
            "Failed inferences by model and status", q("sum by (model, status) (rate(worker_requests_total{%W,%M,status!='OK'}%ri)) > 0"; legend = "{{model}} {{status}}");
            unit = "reqps", w = 16, links = model_links()
        ),
        table(
            "Failures over the range", "model", [Col("Failed", "sum by (model) (increase(worker_requests_total{%W,%M,status!='OK'}%rg)) > 0"; decimals = 0)];
            sort = "Failed", w = 8, h = 8
        ),
    )

    row!(b, "Executable cache"; collapsed = true)
    line!(
        b,
        stat("Cache hits", q("sum(increase(worker_executable_cache_hits_total{%W}%rg))"); w = 4, decimals = 0),
        stat("Cache misses", q("sum(increase(worker_executable_cache_misses_total{%W}%rg))"); w = 4, decimals = 0),
        stat("Load failures", q("sum(increase(worker_executable_cache_failures_total{%W}%rg))"); w = 4, decimals = 0, thresholds = steps("green", (1, "orange"))),
        stat("Programs stored", q("sum(increase(worker_executable_cache_stores_total{%W}%rg))"); w = 4, decimals = 0),
        stat("Compile time", q("sum(increase(worker_executable_cache_compile_seconds_total{%W}%rg))"); w = 4, unit = "s"),
        stat("Cache load time", q("sum(increase(worker_executable_cache_load_seconds_total{%W}%rg))"); w = 4, unit = "s"),
    )
    return b
end

# ---- Model Detail -----------------------------------------------------------------------------------

function model_detail()
    b = Board(
        "rs-model", "ReactantServer: Model Detail",
        "One model: demand, latency, batching, where it is placed, and how often its weights are loaded and evicted.",
        [dsvar(), workervar(), qvar("model", "Model", "label_values(worker_model_max_batch_size, model)"; multi = false, all = false)]
    )
    wm = "%W,%m"
    line!(
        b,
        stat("Calls", q("sum(increase(worker_requests_total{$wm}%rg))"); decimals = 0),
        stat("Failed", q("sum(increase(worker_requests_total{$wm,status!='OK'}%rg)) or vector(0)"); decimals = 0, thresholds = steps("green", (1, "red"))),
        stat("Call rate", q("sum(rate(worker_requests_total{$wm}%ri))"); unit = "reqps", graph = true, decimals = 2),
        stat("p50 latency", q("histogram_quantile(0.5, sum by (le) (increase(worker_request_latency_seconds_bucket{$wm}%rg)))"); unit = "s"),
        stat("p95 latency", q("histogram_quantile(0.95, sum by (le) (increase(worker_request_latency_seconds_bucket{$wm}%rg)))"); unit = "s"),
        stat("p99 latency", q("histogram_quantile(0.99, sum by (le) (increase(worker_request_latency_seconds_bucket{$wm}%rg)))"); unit = "s"),
        stat("Device share", q("sum(increase(worker_compute_seconds_total{$wm}%rg)) / clamp_min(sum(increase(worker_compute_seconds_total{%W}%rg)), 1e-9)"); unit = "percentunit", decimals = 1),
        stat("Per dispatch", q("sum(increase(worker_compute_seconds_total{$wm}%rg)) / clamp_min(sum(increase(worker_dispatch_total{$wm}%rg)), 1)"); unit = "s"),
    )
    line!(
        b,
        stat("Rows per dispatch", q("sum(increase(worker_rows_served_total{$wm}%rg)) / clamp_min(sum(increase(worker_dispatch_total{$wm}%rg)), 1)"); decimals = 2),
        stat("Max batch", q("max(worker_model_max_batch_size{$wm})"); decimals = 0),
        stat("Requests per dispatch", q("sum(increase(worker_requests_served_total{$wm}%rg)) / clamp_min(sum(increase(worker_dispatch_total{$wm}%rg)), 1)"); decimals = 2),
        stat("Weights", q("max(worker_model_weight_bytes{$wm})"); unit = "bytes"),
        stat("Resident on", q("sum(worker_model_resident{$wm})"); decimals = 0, desc = "Workers holding its weights on the device now."),
        stat("Placed on", q("count(gateway_placement_weight{%m,%W} > 0) or vector(0)"); decimals = 0, desc = "Workers the gateway routes it to."),
        stat("Loads over the range", q("sum(increase(worker_model_loads_total{$wm}%rg))"); decimals = 0),
        stat("Mean load time", q("max(worker_model_load_seconds{$wm})"); unit = "s", desc = "Moving average of its weight loads, on the slowest worker."),
    )

    row!(b, "Demand and latency")
    line!(
        b,
        ts("Calls by status", q("sum by (status) (rate(worker_requests_total{$wm}%ri))"; legend = "{{status}}"); unit = "reqps", stack = true),
        ts(
            "Handler latency", [
                q("histogram_quantile(0.5, sum by (le) (rate(worker_request_latency_seconds_bucket{$wm}%ri)))"; legend = "p50"),
                q("histogram_quantile(0.95, sum by (le) (rate(worker_request_latency_seconds_bucket{$wm}%ri)))"; legend = "p95"),
                q("histogram_quantile(0.99, sum by (le) (rate(worker_request_latency_seconds_bucket{$wm}%ri)))"; legend = "p99"),
            ]; unit = "s"
        ),
    )
    line!(
        b,
        heat("Handler latency distribution", "sum by (le) (rate(worker_request_latency_seconds_bucket{$wm}%ri))"),
        ts(
            "Queue", [
                q("sum by (worker) (worker_queue_depth{$wm})"; legend = "{{worker}} queued"),
                q("max by (worker) (worker_queue_wait_seconds{$wm,quantile='0.99'})"; legend = "{{worker}} wait p99"),
            ]; overrides = [by_regex(".*wait p99", "unit" => "s", "custom.axisPlacement" => "right")]
        ),
    )

    row!(b, "Cost and batching")
    line!(
        b,
        ts("Device time per second by worker", q("sum by (worker) (rate(worker_compute_seconds_total{$wm}%ri))"; legend = "{{worker}}"); unit = "percentunit", stack = true, w = 8),
        ts("Rows per dispatch by worker", q("sum by (worker) (rate(worker_rows_served_total{$wm}%ri)) / (sum by (worker) (rate(worker_dispatch_total{$wm}%ri)) > 0)"; legend = "{{worker}}"); w = 8),
        ts(
            "Rows, requests and dispatches", [
                q("sum(rate(worker_rows_served_total{$wm}%ri))"; legend = "rows"),
                q("sum(rate(worker_requests_served_total{$wm}%ri))"; legend = "requests"),
                q("sum(rate(worker_dispatch_total{$wm}%ri))"; legend = "dispatches"),
            ]; unit = "ops", w = 8
        ),
    )

    row!(b, "Placement and weights")
    line!(
        b,
        timeline(
            "Resident on the device, by worker", q("max by (worker) (worker_model_resident{$wm})"; legend = "{{worker}}");
            mappings = [D("type" => "value", "options" => D("0" => D("text" => "off", "color" => "transparent", "index" => 0), "1" => D("text" => "resident", "color" => "green", "index" => 1)))],
            w = 12, h = 7
        ),
        ts("Gateway placement weight by worker", q("max by (worker) (gateway_placement_weight{%m,%W})"; legend = "{{worker}}"); w = 12, h = 7, max = 1, stack = true),
    )
    line!(
        b,
        ts(
            "Loads and evictions per minute, by worker", [
                q(per_min("sum by (worker) (rate(worker_model_loads_total{$wm}%ri))"); legend = "{{worker}} loaded"),
                q(per_min("sum by (worker) (rate(worker_model_reads_total{$wm}%ri))"); legend = "{{worker}} read the file"),
                q(per_min("sum by (worker) (rate(worker_model_evicts_total{$wm}%ri))"); legend = "{{worker}} evicted"),
            ]; w = 12, overrides = [by_regex(".*evicted", "custom.transform" => "negative-Y")],
            desc = "Evictions plot below the axis. A model that is loaded and evicted over and over is thrashing: its placement or the budget is too tight for its traffic."
        ),
        ts(
            "Weight load time (moving average) by worker", q("max by (worker) (worker_model_load_seconds{$wm} > 0)"; legend = "{{worker}}");
            unit = "s", w = 12
        ),
    )
    return b
end

# ---- Memory and Weights ---------------------------------------------------------------------------

const MEMORY_HELP = """
Each worker's device memory is one **arena** run by XLA's BFC allocator. It holds the **weights** of the resident models and the **scratch** that executions use while they run. Admission keeps one free block at least as large as the largest scratch of any resident model (the **scratch reserve**), so every resident model can still run.

**Fragmentation** is 1 minus the largest free block over the free bytes: 0 when the free memory is one block, near 1 when it is splintered. Fragmentation matters when the largest free block cannot hold the next model's weights plus the scratch reserve, even though enough bytes are free.

**Eviction reasons**: *bytes*, too few free bytes (or the weight budget is full); *contiguity*, enough bytes but no block that holds the weights, which is fragmentation costing a load; *scratch*, evicted after an upload to keep the scratch reserve free. A **fallback compaction** frees every evictable model and reloads the valuable ones packed together, when eviction alone cannot make room.

A load copies the weights to the device (**upload**) after reading them into host memory: from a host copy when one is cached, otherwise from the weights **file**. With a host weight budget, reads for queued models run **ahead of the dispatch** on a background task so the worker keeps serving meanwhile.
"""

function memory()
    b = Board(
        "rs-memory", "ReactantServer: Memory and Weights",
        "Device memory, fragmentation, and the cost of moving weights: loads, evictions and why, file reads, the host weight cache.",
        [dsvar(), workervar(), modelvar(), topnvar()]
    )
    lf = "worker_device_memory_largest_free_block_bytes{%W}"
    scratch = "worker_weight_cache_max_scratch_bytes{%W}"
    line!(
        b,
        stat(
            "Arena in use, fullest worker", q("max(worker_device_memory_in_use_bytes{%W} / worker_device_memory_limit_bytes{%W})");
            unit = "percentunit", decimals = 1, thresholds = steps("green", (0.85, "orange"), (0.95, "red"))
        ),
        stat(
            "Fragmentation, worst worker", q("max(worker_device_memory_fragmentation{%W})"); unit = "percentunit", decimals = 0,
            thresholds = steps("green", (0.3, "orange"), (0.6, "red"))
        ),
        stat(
            "Contiguous headroom, tightest", q("min($lf - on (worker) $scratch)"); unit = "bytes",
            thresholds = steps("red", (0, "orange"), (1.073741824e9, "green")),
            desc = "Largest free block minus the scratch reserve: the largest model that fits now without an eviction for contiguity."
        ),
        stat("Weight loads over the range", q("sum(increase(worker_weight_loads_total{%W}%rg))"); decimals = 0),
        stat("Read from weights files", q("sum(increase(worker_weight_read_bytes_total{%W}%rg))"); unit = "bytes"),
        stat(
            "Host copy hit ratio", q("sum(increase(worker_host_weight_hits_total{%W}%rg)) / clamp_min(sum(increase(worker_host_weight_hits_total{%W}%rg)) + sum(increase(worker_host_weight_misses_total{%W}%rg)), 1)");
            unit = "percentunit", decimals = 0, thresholds = steps("red", (0.5, "orange"), (0.9, "green")),
            desc = "Loads served from a cached host copy over the range. Empty without a host weight budget."
        ),
        stat(
            "Evictions for contiguity", q("sum(increase(worker_weight_evicts_total{%W,reason='contiguity'}%rg))"); decimals = 0,
            thresholds = steps("green", (1, "orange")), desc = "Over the range: models evicted although enough bytes were free, because no single block held the new weights."
        ),
        stat(
            "Admission failures", q("sum(increase(worker_weight_admission_failures_total{%W}%rg))"); decimals = 0,
            thresholds = steps("green", (1, "red")), desc = "Over the range: models that could not be placed even after compacting. Their requests failed."
        ),
    )

    row!(b, "Device memory")
    line!(
        b,
        ts(
            "Arena: \$worker", [
                q("sum(worker_weight_cache_resident_bytes{%W}) + sum(worker_weight_cache_pinned_bytes{%W})"; legend = "weights"),
                q("clamp_min(sum(worker_device_memory_in_use_bytes{%W}) - sum(worker_weight_cache_resident_bytes{%W}) - sum(worker_weight_cache_pinned_bytes{%W}), 0)"; legend = "scratch and other"),
                q("sum($lf)"; legend = "largest free block"),
                q("sum(worker_device_memory_free_bytes{%W}) - sum($lf)"; legend = "free in smaller blocks"),
                q("sum($scratch)"; legend = "scratch reserve"),
            ]; unit = "bytes", stack = true, legend = :list, w = 6, h = 9,
            overrides = [
                dashed("scratch reserve"),
                by_name("weights", "color" => D("mode" => "fixed", "fixedColor" => "blue")),
                by_name("scratch and other", "color" => D("mode" => "fixed", "fixedColor" => "purple")),
                by_name("largest free block", "color" => D("mode" => "fixed", "fixedColor" => "green")),
                by_name("free in smaller blocks", "color" => D("mode" => "fixed", "fixedColor" => "orange")),
                by_name("scratch reserve", "color" => D("mode" => "fixed", "fixedColor" => "red")),
            ],
            desc = "The arena, stacked to its limit. Orange is free memory no large allocation can use; when green falls below the red line, the next load needs an eviction for contiguity.",
            extra = D("repeat" => "worker", "repeatDirection" => "h", "maxPerRow" => 4)
        ),
    )
    line!(
        b,
        ts(
            "Weights on the device against their budget", [
                q("sum by (worker) (worker_weight_cache_resident_bytes{%W})"; legend = "{{worker}} on demand"),
                q("sum by (worker) (worker_weight_cache_max_bytes{%W})"; legend = "{{worker}} budget"),
            ]; unit = "bytes", w = 8, overrides = [dashed(".*budget")],
            desc = "Device-pinned weights are outside this budget."
        ),
        ts("Peak arena use since start", q("max by (worker) (worker_device_memory_peak_in_use_bytes{%W} / worker_device_memory_limit_bytes{%W})"; legend = "{{worker}}"); unit = "percentunit", max = 1, w = 8),
        ts(
            "Device memory outside the arena", q("max by (worker) (worker_device_memory_out_of_pool_bytes{%W})"; legend = "{{worker}}"); unit = "bytes", w = 8,
            desc = "Driver memory the allocator cannot see: the CUDA context, loaded modules, command buffers. It competes for the headroom between the arena and the card."
        ),
    )

    row!(b, "Fragmentation")
    line!(
        b,
        ts(
            "Fragmentation by worker", q("max by (worker) (worker_device_memory_fragmentation{%W})"; legend = "{{worker}}"); unit = "percentunit", min = 0, max = 1, w = 8,
            thresholds = steps("green", (0.6, "red")), custom = D("thresholdsStyle" => D("mode" => "dashed"))
        ),
        ts(
            "Largest free block and the scratch reserve", [
                q("max by (worker) ($lf)"; legend = "{{worker}} largest free block"),
                q("max by (worker) ($scratch)"; legend = "{{worker}} scratch reserve"),
            ]; unit = "bytes", w = 8, overrides = [dashed(".*scratch reserve")],
            desc = "What is left above each dashed line is the largest model that loads without an eviction for contiguity."
        ),
        ts(
            "Evictions per minute, by reason", q(per_min("sum by (reason) (rate(worker_weight_evicts_total{%W}%ri))"); legend = "{{reason}}");
            stack = true, bars = true, w = 8, overrides = [by_name("contiguity", "color" => D("mode" => "fixed", "fixedColor" => "orange")), by_name("scratch", "color" => D("mode" => "fixed", "fixedColor" => "purple")), by_name("bytes", "color" => D("mode" => "fixed", "fixedColor" => "blue"))]
        ),
    )
    line!(
        b,
        ts(
            "Compactions per minute", [
                q(per_min("sum(rate(worker_weight_compactions_total{%W}%ri))"); legend = "compactions"),
                q(per_min("sum(rate(worker_weight_fallback_compactions_total{%W}%ri))"); legend = "of which admission fallbacks"),
                q(per_min("sum(rate(worker_weight_admission_failures_total{%W}%ri))"); legend = "admission failures"),
            ]; w = 12, overrides = [by_name("admission failures", "color" => D("mode" => "fixed", "fixedColor" => "red"))]
        ),
        ts(
            "Time spent compacting, per second", q("sum by (worker) (rate(worker_weight_compaction_seconds_total{%W}%ri))"; legend = "{{worker}}");
            unit = "percentunit", w = 12, desc = "The dispatch loop serves nothing while a compaction frees and reloads weights."
        ),
    )

    row!(b, "Weight loading")
    line!(
        b,
        ts(
            "Loads and evictions per minute, by worker", [
                q(per_min("sum by (worker) (rate(worker_weight_loads_total{%W}%ri))"); legend = "{{worker}} loaded"),
                q(per_min("sum by (worker) (rate(worker_weight_evicts_total{%W}%ri))"); legend = "{{worker}} evicted"),
            ]; w = 12, overrides = [by_regex(".*evicted", "custom.transform" => "negative-Y")], desc = "Evictions plot below the axis."
        ),
        ts(
            "Load time per second of wall time", [
                q("sum by (worker) (rate(worker_weight_upload_seconds_total{%W}%ri))"; legend = "{{worker}} upload"),
                q("sum by (worker) (rate(worker_weight_read_seconds_total{%W}%ri))"; legend = "{{worker}} file read"),
            ]; unit = "percentunit", stack = true, w = 12,
            desc = "Uploads hold the worker's dispatch loop. File reads ran either on the loop or, read ahead of the dispatch, on a background task; Reading ahead below shows the background share."
        ),
    )
    line!(
        b,
        ts(
            "Bandwidth: upload and file read", [
                q("sum by (worker) (rate(worker_weight_load_bytes_total{%W}[10m])) / (sum by (worker) (rate(worker_weight_upload_seconds_total{%W}[10m])) > 0)"; legend = "{{worker}} upload"),
                q("sum by (worker) (rate(worker_weight_read_bytes_total{%W}[10m])) / (sum by (worker) (rate(worker_weight_read_seconds_total{%W}[10m])) > 0)"; legend = "{{worker}} file read"),
            ]; unit = "Bps", w = 8, desc = "Over 10-minute windows, so gaps mean no loads. Admission prices an unmeasured model's reload with these rates."
        ),
        ts(
            "Mean time per load", q("sum by (worker) (rate(worker_weight_load_seconds_total{%W}[10m])) / (sum by (worker) (rate(worker_weight_loads_total{%W}[10m])) > 0)"; legend = "{{worker}}");
            unit = "s", w = 8, desc = "End to end, over 10-minute windows, including any read done ahead of the dispatch."
        ),
        ts(
            "Host weight cache", [
                q("max(worker_host_weight_cached_bytes{%W})"; legend = "cached"),
                q("max(worker_host_weight_budget_bytes{%W})"; legend = "budget"),
            ]; unit = "bytes", w = 8, overrides = [dashed("budget")],
            desc = "With shared host weights the cache and budget are node-wide, so every worker reports the same numbers."
        ),
    )
    line!(
        b,
        ts(
            "Reading ahead of the dispatch", [
                q("sum by (state) (worker_weight_staging_models{%W})"; legend = "{{state}}"),
            ]; stack = true, w = 8, desc = "Models whose weights are being read (reading) or are read and waiting for their turn on the device (staged)."
        ),
        ts(
            "Background read time per second", [
                q("sum by (worker) (rate(worker_weight_stage_seconds_total{%W}%ri))"; legend = "{{worker}}"),
            ]; unit = "percentunit", w = 8, desc = "Shared-store lock waits included. The dispatch loop keeps serving other models during this time."
        ),
        ts(
            "Host copy hit ratio by worker",
            q("sum by (worker) (rate(worker_host_weight_hits_total{%W}[10m])) / (sum by (worker) (rate(worker_host_weight_hits_total{%W}[10m]) + rate(worker_host_weight_misses_total{%W}[10m])) > 0)"; legend = "{{worker}}");
            unit = "percentunit", min = 0, max = 1, w = 8
        ),
    )

    row!(b, "Churn by model")
    line!(
        b,
        table(
            "Weight churn over the range", "model", [
                Col("Loads", "sum by (model) (increase(worker_model_loads_total{%W,%M}%rg))"; decimals = 0),
                Col("File reads", "sum by (model) (increase(worker_model_reads_total{%W,%M}%rg))"; decimals = 0),
                Col("Evictions", "sum by (model) (increase(worker_model_evicts_total{%W,%M}%rg))"; decimals = 0),
                Col("Bytes loaded", "sum by (model) (increase(worker_model_loads_total{%W,%M}%rg) * worker_model_weight_bytes{%W,%M})"; unit = "bytes"),
                Col("Weights", "max by (model) (worker_model_weight_bytes{%W,%M})"; unit = "bytes"),
                Col("Mean load time", "max by (model) (worker_model_load_seconds{%W,%M} > 0)"; unit = "s"),
                Col("Scratch", "max by (model) (worker_model_scratch_bytes{%W,%M} > 0)"; unit = "bytes"),
                Col("Calls", "sum by (model) (increase(worker_dispatch_total{%W,%M}%rg))"; decimals = 0),
                Col("Resident on", "sum by (model) (worker_model_resident{%W,%M})"; decimals = 0),
            ]; sort = "Bytes loaded", w = 24, h = 12,
            desc = "Calls here are dispatches. Many loads for few dispatches means the model is evicted between uses: the budget or the placement is too tight for it."
        ),
    )
    line!(
        b,
        ts("Loads per minute, top N models", q(per_min(top("sum by (model) (rate(worker_model_loads_total{%W,%M}%ri))", "worker_model_loads_total")); legend = "{{model}}"); stack = true, links = model_links()),
        ts("Evictions per minute, top N models", q(per_min(top("sum by (model) (rate(worker_model_evicts_total{%W,%M}%ri))", "worker_model_evicts_total")); legend = "{{model}}"); stack = true, links = model_links()),
    )
    return b
end

# ---- Placement and Routing ------------------------------------------------------------------------

const PLACEMENT_HELP = """
With `scheduling.mode: lpt_packing` the gateway decides which workers serve each model (the **placement**) and routes each request to one of them. A **repack** revises the placement, at startup and when enough compute has passed; models it moves must be loaded on their new worker, so watch loads and file reads after one (repacks are marked on every graph).

**Planned weights** are the bytes of the models placed on a worker, against its on-demand weight budget. Above 100% the worker cannot hold all of them and will evict among them as traffic shifts.

**Work spread** is the coefficient of variation of device busy time across workers: 0 means even. A standing spread with low planned imbalance means routing is not following the plan; compare the planned and actual shares.
"""

function placement()
    b = Board(
        "rs-placement", "ReactantServer: Placement and Routing",
        "Where the gateway places models, what its repacks move, and whether work is spread evenly across the workers.",
        [dsvar(), workervar(), modelvar(), topnvar()]
    )
    planned = "max by (worker) (gateway_worker_assigned_weight_bytes{%W}) / (max by (worker) (gateway_worker_weight_budget_bytes{%W}) > 0)"
    line!(
        b,
        stat("Repacks over the range", q("sum(increase(gateway_repacks_total%rg))"); decimals = 0),
        stat("Models moved", q("sum(increase(gateway_repack_moved_models_total%rg))"); decimals = 0),
        stat("Weights moved", q("sum(increase(gateway_repack_moved_bytes_total%rg))"); unit = "bytes"),
        stat(
            "Planned weights, fullest worker", q("max($planned)"); unit = "percentunit", decimals = 0,
            thresholds = steps("green", (0.9, "orange"), (1.0, "red")), desc = "Assigned weight bytes over the worker's budget, from the last repack."
        ),
        stat(
            "Work spread", q("stddev($BUSY) / clamp_min(avg($BUSY), 1e-9)"); unit = "percentunit", decimals = 0,
            thresholds = steps("green", (0.1, "orange"), (0.25, "red"))
        ),
        stat("Busiest worker", q("max($BUSY)"); unit = "percentunit", decimals = 1, thresholds = steps("green", (0.7, "orange"), (0.9, "red"))),
        stat(
            "In-flight spread", q("max(sum by (worker) (gateway_worker_inflight_work{%W})) - min(sum by (worker) (gateway_worker_inflight_work{%W}))");
            decimals = 2, desc = "Busiest minus idlest worker, in the routing basis' units."
        ),
        stat("Models placed", q("count(max by (model) (gateway_placement_weight) > 0) or vector(0)"); decimals = 0),
    )
    line!(b, text("How placement works", PLACEMENT_HELP; h = 5))

    row!(b, "Planned weights against budget")
    line!(
        b,
        bars("Planned weights over budget, by worker", q(planned; legend = "{{worker}}", instant = true); unit = "percentunit", min = 0, max = 1.25, w = 8, thresholds = steps("green", (0.9, "orange"), (1.0, "red")), color = D("mode" => "thresholds")),
        ts(
            "Planned weights and budget", [
                q("max by (worker) (gateway_worker_assigned_weight_bytes{%W})"; legend = "{{worker}} planned"),
                q("max by (worker) (gateway_worker_weight_budget_bytes{%W})"; legend = "{{worker}} budget"),
                q("sum by (worker) (worker_weight_cache_resident_bytes{%W})"; legend = "{{worker}} on the device"),
            ]; unit = "bytes", w = 16, overrides = [dashed(".*budget"), by_regex(".*on the device", "custom.lineStyle" => D("fill" => "dot", "dash" => [0, 10]))],
            desc = "Planned is what the last repack assigned; on the device is what is resident now. Both step at a repack."
        ),
    )

    row!(b, "Repacks")
    line!(
        b,
        ts("Repacks per hour, by trigger", q("sum by (trigger) (rate(gateway_repacks_total[30m])) * 3600"; legend = "{{trigger}}"); w = 8, stack = true, desc = "Over 30-minute windows."),
        ts(
            "Moved by repacks, per hour", [
                q("sum(rate(gateway_repack_moved_models_total[30m])) * 3600"; legend = "models"),
                q("sum(rate(gateway_repack_moved_bytes_total[30m])) * 3600"; legend = "weights"),
            ]; w = 8, overrides = [by_name("weights", "unit" => "bytes", "custom.axisPlacement" => "right")], desc = "Over 30-minute windows."
        ),
        ts(
            "Loads that followed, per minute", [
                q(per_min("sum by (worker) (rate(worker_weight_loads_total{%W}%ri))"); legend = "{{worker}} loads"),
                q(per_min("sum by (worker) (rate(worker_host_weight_misses_total{%W}%ri))"); legend = "{{worker}} file reads"),
            ]; w = 8, overrides = [dashed(".*file reads")]
        ),
    )

    row!(b, "Load balance")
    line!(
        b,
        ts("Device busy by worker", [q(BUSY; legend = "{{worker}}"), q("avg($BUSY)"; legend = "mean")]; unit = "percentunit", max = 1, overrides = [dashed("mean")], w = 8),
        ts(
            "Planned and actual share of device work", [
                q("sum by (worker) (gateway_placement_weight{%W} * on (model) group_left () gateway_model_utilization) / scalar(clamp_min(sum(gateway_model_utilization), 1e-9))"; legend = "{{worker}} planned"),
                q("$BUSY / scalar(clamp_min(sum(rate(worker_compute_seconds_total%ri)), 1e-9))"; legend = "{{worker}} actual"),
            ]; unit = "percentunit", w = 8, overrides = [dashed(".*planned")],
            desc = "Planned is the packer's intent: placement weight times expected utilization. A gap between the lines means routing is not following the plan."
        ),
        ts(
            "Work spread", [
                q("stddev($BUSY) / clamp_min(avg($BUSY), 1e-9)"; legend = "spread"),
            ]; unit = "percentunit", w = 8, thresholds = steps("green", (0.25, "red")), custom = D("thresholdsStyle" => D("mode" => "dashed")),
            desc = "A spike during a burst is normal; a flat elevated line is a standing imbalance."
        ),
    )
    line!(
        b,
        ts("In-flight work by worker", q("sum by (worker) (gateway_worker_inflight_work{%W})"; legend = "{{worker}}"); w = 12),
        ts("Requests routed per worker", q("sum by (worker) (rate(gateway_replica_routed_total{%W}%ri))"; legend = "{{worker}}"); unit = "reqps", stack = true, w = 12),
    )

    row!(b, "Placement")
    line!(
        b,
        panel(
            "table", "Placement weight, model by worker", [q("max by (model, worker) (gateway_placement_weight{%W,%M}) > 0"; instant = true, fmt = "table")];
            desc = "The share of a model's traffic each worker receives. A row with several workers is a replicated model.",
            w = 14, h = 12, decimals = 2, custom = D("align" => "auto", "filterable" => true, "cellOptions" => D("type" => "color-text"), "inspect" => false),
            options = D("showHeader" => true, "cellHeight" => "sm"),
            overrides = [by_name("model", "links" => [D("title" => "Model detail: \${__value.raw}", "url" => MODEL_CELL_LINK)])],
            extra = D("transformations" => [D("id" => "groupingToMatrix", "options" => D("columnField" => "worker", "rowField" => "model", "valueField" => "Value", "emptyValue" => "empty"))])
        ),
        bars(
            "Expected utilization, top N", q("topk(%n, max by (model) (gateway_model_utilization{%M}))"; legend = "{{model}}", instant = true);
            unit = "percentunit", w = 10, h = 12, links = model_links(),
            desc = "Arrival rate times compute cost: the device time per second the packer expects each model to need."
        ),
    )

    row!(b, "Gateway to worker")
    line!(
        b,
        ts(
            "Worker call latency p99", q("histogram_quantile(0.99, sum by (worker, le) (rate(gateway_worker_latency_seconds_bucket{rpc='ModelInfer',%W}%ri)))"; legend = "{{worker}}");
            unit = "s", w = 8
        ),
        ts(
            "Health probe round trip, p99", q("histogram_quantile(0.99, sum by (worker, le) (rate(gateway_worker_probe_seconds_bucket{%W}%ri)))"; legend = "{{worker}}");
            unit = "s", w = 8, thresholds = steps("green", (5, "red")), custom = D("thresholdsStyle" => D("mode" => "dashed"))
        ),
        timeline("Worker ready", q("max by (worker) (gateway_worker_ready{%W})"; legend = "{{worker}}"); w = 8),
    )
    return b
end

# ---- Resources ------------------------------------------------------------------------------------

const RESOURCES_HELP = """
**What this dashboard cannot show.** *Device busy* is time the worker spent inside model executions, timed on the host; it is not SM occupancy, memory bandwidth or power, which need a GPU exporter such as DCGM. Host memory pressure, swap and `/dev/shm` use outside the host weight cache are not exported; nor is a stall of a worker's dispatch thread directly, though the health probe round trip (Overview) is the earliest sign of one. Series with no `worker` label are the gateway's own process.
"""

function resources()
    b = Board(
        "rs-resources", "ReactantServer: Resources",
        "Processes, Julia GC, I/O, device memory held by each worker, and the health of the monitoring itself.",
        [dsvar(), jobvar(), workervar()]
    )
    line!(b, text("What this dashboard cannot show", RESOURCES_HELP; h = 4))
    gw = "%job,worker=''"
    row!(b, "Device")
    line!(
        b,
        bars("Arena in use", q("max by (worker) (worker_device_memory_in_use_bytes{%W} / worker_device_memory_limit_bytes{%W})"; legend = "{{worker}}", instant = true); unit = "percentunit", min = 0, max = 1, w = 8, h = 7),
        ts(
            "Device memory held by the worker", [
                q("max by (worker) (worker_device_memory_process_used_bytes{%W})"; legend = "{{worker}} process"),
                q("max by (worker) (worker_device_memory_limit_bytes{%W})"; legend = "{{worker}} arena"),
            ]; unit = "bytes", w = 8, h = 7, overrides = [dashed(".*arena")], desc = "Per the driver, as nvidia-smi reports it, against the arena the allocator owns."
        ),
        ts("Device busy by worker", q(BUSY; legend = "{{worker}}"); unit = "percentunit", max = 1, w = 8, h = 7),
    )
    row!(b, "CPU and memory")
    line!(
        b,
        ts(
            "Process CPU", [
                q("sum by (worker) (rate(process_cpu_seconds_total{%W}%ri))"; legend = "{{worker}}"),
                q("sum(rate(process_cpu_seconds_total{$gw}%ri))"; legend = "gateway"),
            ]; unit = "percentunit", desc = "Cores in use (1 = one core)."
        ),
        ts(
            "Process resident memory", [
                q("max by (worker) (process_resident_memory_bytes{%W})"; legend = "{{worker}}"),
                q("max(process_resident_memory_bytes{$gw})"; legend = "gateway"),
            ]; unit = "bytes"
        ),
    )
    line!(
        b,
        ts(
            "Open file descriptors", [
                q("max by (worker) (process_open_fds{%W})"; legend = "{{worker}}"),
                q("max(process_open_fds{$gw})"; legend = "gateway"),
            ]; w = 8
        ),
        ts(
            "Virtual memory", [
                q("max by (worker) (process_virtual_memory_bytes{%W})"; legend = "{{worker}}"),
                q("max(process_virtual_memory_bytes{$gw})"; legend = "gateway"),
            ]; unit = "bytes", w = 8
        ),
        ts(
            "Uptime", [
                q("time() - max by (worker) (process_start_time_seconds{%W})"; legend = "{{worker}}"),
                q("time() - max(process_start_time_seconds{$gw})"; legend = "gateway"),
            ]; unit = "s", w = 8, legend = :list
        ),
    )
    row!(b, "Julia GC")
    line!(
        b,
        ts(
            "GC time share", [
                q("sum by (worker) (rate(julia_gc_seconds_total{%W}%ri))"; legend = "{{worker}}"),
                q("sum(rate(julia_gc_seconds_total{$gw}%ri))"; legend = "gateway"),
            ]; unit = "percentunit", w = 8, desc = "Share of wall time spent in garbage collection."
        ),
        ts(
            "Collections per minute", [
                q(per_min("sum by (worker, type) (rate(julia_gc_collections_total{%W}%ri))"); legend = "{{worker}} {{type}}"),
                q(per_min("sum by (type) (rate(julia_gc_collections_total{$gw}%ri))"); legend = "gateway {{type}}"),
            ]; w = 8
        ),
        ts(
            "Live heap", [
                q("max by (worker) (julia_gc_live_bytes{%W})"; legend = "{{worker}}"),
                q("max(julia_gc_live_bytes{$gw})"; legend = "gateway"),
            ]; unit = "bytes", w = 8
        ),
    )
    line!(
        b,
        ts(
            "Allocation rate", [
                q("sum by (worker) (rate(julia_gc_alloc_bytes_total{%W}%ri))"; legend = "{{worker}}"),
                q("sum(rate(julia_gc_alloc_bytes_total{$gw}%ri))"; legend = "gateway"),
            ]; unit = "Bps", w = 24, h = 7
        ),
    )
    row!(b, "I/O")
    line!(
        b,
        ts(
            "Storage reads and writes", [
                q("sum by (worker) (rate(process_io_read_bytes_total{%W}%ri))"; legend = "{{worker}} read"),
                q("sum by (worker) (rate(process_io_write_bytes_total{%W}%ri))"; legend = "{{worker}} write"),
            ]; unit = "Bps", desc = "Bytes that reached the block layer: weights files read past the page cache show here."
        ),
        ts(
            "All reads and writes, page cache and sockets included", [
                q("sum by (worker) (rate(process_io_rchar_bytes_total{%W}%ri))"; legend = "{{worker}} read"),
                q("sum by (worker) (rate(process_io_wchar_bytes_total{%W}%ri))"; legend = "{{worker}} write"),
            ]; unit = "Bps"
        ),
    )
    row!(b, "Monitoring")
    line!(
        b,
        timeline("Scrape target up", q("max by (job) (up{%job})"; legend = "{{job}}"); w = 8, h = 6),
        timeline("Worker metrics in the last scrape", q("max by (worker) (gateway_worker_metrics_up{%W})"; legend = "{{worker}}"); w = 8, h = 6),
        ts(
            "Scrape duration", q("max by (job) (scrape_duration_seconds{%job})"; legend = "{{job}}"); unit = "s", w = 8, h = 6, legend = :list,
            desc = "One scrape fans out to every worker's metrics endpoint, so a slow or stalled worker lengthens it."
        ),
    )
    return b
end

# ---- Main -----------------------------------------------------------------------------------------

const BOARDS = [
    "overview.json" => overview, "models.json" => models, "model_detail.json" => model_detail,
    "memory.json" => memory, "placement.json" => placement, "resources.json" => resources,
]

# Every expression of a dashboard, panels in collapsed rows and annotations included.
function exprs(board)
    out = Tuple{String, String}[]
    walk(ps) = for pp in ps
        haskey(pp, "panels") && walk(pp["panels"])
        for t in get(pp, "targets", [])
            push!(out, (get(pp, "title", ""), t["expr"]))
        end
    end
    walk(board["panels"])
    for a in board["annotations"]["list"]
        haskey(a, "expr") && push!(out, ("annotation: " * a["name"], a["expr"]))
    end
    return out
end

# Substitute the variables the way Grafana would with everything selected.
function concrete(expr)
    return replace(
        expr, "\$__rate_interval" => "1m", "\$__range" => "1h", "\$topn" => "10",
        "\$worker" => ".*", "\$model" => ".+", "\$job" => ".+",
    )
end

function check(prom)
    errors = 0
    empty = 0
    for (file, f) in BOARDS
        board = render(f())
        for (title, expr) in exprs(board)
            e = concrete(expr)
            # A one-sample model on Model Detail: pick any model that exists.
            occursin("model='\$model'", expr) && (e = replace(concrete(replace(expr, "model='\$model'" => "model=~'.+'"))))
            url = prom * "/api/v1/query?query=" * _escape(e)
            body = try
                String(take!(Downloads.download(url, IOBuffer())))
            catch err
                err isa Downloads.RequestError && err.response.status >= 400 ? err.message * " " * sprint(showerror, err) : rethrow()
            end
            if !occursin("\"status\":\"success\"", body)
                errors += 1
                println("ERROR  $file / $title\n       $e\n       $(first(body, 300))")
            elseif occursin("\"result\":[]", body)
                empty += 1
                println("empty  $file / $title")
            end
        end
    end
    println("$errors errors, $empty empty")
    return errors == 0
end

_escape(s) = join(c in 'A':'Z' || c in 'a':'z' || c in '0':'9' || c in "-_.~" ? string(c) : join("%" * uppercase(string(b; base = 16, pad = 2)) for b in codeunits(string(c))) for c in s)

function main(args)
    if length(args) == 2 && args[1] == "--check"
        exit(check(rstrip(args[2], '/')) ? 0 : 1)
    end
    isempty(args) || (println(stderr, "usage: build.jl [--check PROMETHEUS_URL]"); exit(2))
    mkpath(OUT)
    for (file, f) in BOARDS
        io = IOBuffer()
        json(io, render(f()))
        println(io)
        write(joinpath(OUT, file), take!(io))
        println("wrote ", joinpath("dashboards", file))
    end
    return
end

main(ARGS)
