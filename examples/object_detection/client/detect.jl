# Step 3 of 3: send an image to the running server and draw the detections.
#
#   julia --project=examples/object_detection/client examples/object_detection/client/detect.jl [IMAGE_PATH]
#
# Connects to the server from step 2 on 127.0.0.1:$OD_PORT (default 8080). With no IMAGE_PATH it uses
# (or downloads) a default object-rich photo. Writes examples/object_detection/detections.jpg.
# This environment has no Reactant and no PythonCall, so it loads fast.

using Downloads
using FileIO
using ImageTransformations: imresize
using Colors
using CairoMakie
using ReactantServerClient

const HERE = @__DIR__
const ASSETS = normpath(joinpath(HERE, ".."))      # shared test image / output live next to the envs
const MODEL = "object_detector"
const IMG_SIZE = 640                                # must match the export image_size
const DISPLAY_THRESH = 0.5
const HOST = get(ENV, "OD_HOST", "127.0.0.1")
const PORT = parse(Int, get(ENV, "OD_PORT", "8080"))
const DEFAULT_IMAGE_URL = "https://ultralytics.com/images/bus.jpg"

# torchvision FasterRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"] (index = class id; "N/A" are
# unused COCO ids). A detection's class id `c` (1..90) names COCO_LABELS[c + 1].
const COCO_LABELS = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A", "backpack",
    "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

label_for(class_id::Integer) =
    (1 <= class_id + 1 <= length(COCO_LABELS)) ? COCO_LABELS[class_id + 1] : "class $class_id"

# Resolve the image path: explicit arg, else (download) the default shared test image.
function resolve_image()
    arg = findfirst(a -> !startswith(a, "--"), ARGS)
    arg === nothing || return ARGS[arg]
    dest = joinpath(ASSETS, "test_image.jpg")
    isfile(dest) || (@info "Downloading test image" url = DEFAULT_IMAGE_URL;
                     Downloads.download(DEFAULT_IMAGE_URL, dest))
    return dest
end

# Load + resize to the network input; returns (input::Array{UInt8,4} (W,H,C,1), display::Matrix{RGB}).
function load_image(path)
    img = RGB.(FileIO.load(path))                 # normalize Gray/RGBA -> RGB, indexed [y, x]
    img = imresize(img, (IMG_SIZE, IMG_SIZE))     # plain resize (no aspect pad)
    arr = Array{UInt8}(undef, IMG_SIZE, IMG_SIZE, 3, 1)   # (W,H,C,N) -> wire (1,3,H,W)
    @inbounds for y in 1:IMG_SIZE, x in 1:IMG_SIZE
        px = img[y, x]
        arr[x, y, 1, 1] = round(UInt8, clamp(float(red(px)) * 255, 0, 255))
        arr[x, y, 2, 1] = round(UInt8, clamp(float(green(px)) * 255, 0, 255))
        arr[x, y, 3, 1] = round(UInt8, clamp(float(blue(px)) * 255, 0, 255))
    end
    return arr, img
end

# OUTPUT__0 is [x1,y1,x2,y2,score,class] per detection; the meta emits (6, Ndet). Detect orientation.
function parse_detections(out)
    ndims(out) == 2 || (out = reshape(out, length(out) ÷ 6, 6))
    dets = size(out, 1) == 6 ? out : permutedims(out)
    [(; x1 = Float64(dets[1, j]), y1 = Float64(dets[2, j]), x2 = Float64(dets[3, j]),
       y2 = Float64(dets[4, j]), score = Float64(dets[5, j]), class = round(Int, dets[6, j]))
     for j in 1:size(dets, 2)]
end

function draw(display_img, dets, outfile)
    fig = Figure(size = (IMG_SIZE, IMG_SIZE))
    ax = Axis(fig[1, 1]; aspect = DataAspect(), yreversed = true)
    hidedecorations!(ax); hidespines!(ax)
    image!(ax, 0 .. IMG_SIZE, 0 .. IMG_SIZE, permutedims(display_img))
    shown = 0
    for d in dets
        d.score >= DISPLAY_THRESH || continue
        shown += 1
        lines!(ax, [d.x1, d.x2, d.x2, d.x1, d.x1], [d.y1, d.y1, d.y2, d.y2, d.y1];
               color = :red, linewidth = 2)
        text!(ax, d.x1 + 2, d.y1 + 2; text = "$(label_for(d.class)) $(round(d.score; digits = 2))",
              color = :yellow, fontsize = 14, align = (:left, :top))
    end
    # CairoMakie's save() only writes Cairo formats; render to a buffer and write the jpeg via FileIO.
    if lowercase(splitext(outfile)[2]) in (".png", ".svg", ".pdf", ".eps")
        save(outfile, fig)
    else
        FileIO.save(outfile, RGB.(colorbuffer(fig)))   # RGB. drops alpha (jpeg has no alpha)
    end
    return shown
end

function main()
    image_path = resolve_image()
    @info "Loading image" image_path
    input, display_img = load_image(image_path)

    kserve_init()
    try
        model = KServeModel("grpc://$HOST:$PORT", MODEL; max_batch_size = 1)
        @info "Running inference" server = "$HOST:$PORT"
        resp = infer_sync(model, [InferInput("INPUT__0", input)])
        out = InferOutput("OUTPUT__0", resp, Float32)
        @info "Raw output" size = size(out)
        dets = parse_detections(out)
        outfile = joinpath(ASSETS, "detections.jpg")
        shown = draw(display_img, dets, outfile)
        @info "Done" total_detections = length(dets) drawn = shown output = outfile
        for d in sort(dets; by = x -> -x.score)
            d.score >= DISPLAY_THRESH || continue
            println("  $(label_for(d.class))  score=$(round(d.score; digits=3))  " *
                    "box=($(round(d.x1)), $(round(d.y1)), $(round(d.x2)), $(round(d.y2)))")
        end
    finally
        kserve_shutdown()
    end
end

main()
