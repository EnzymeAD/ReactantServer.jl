# DetectionGlue convention tests. Pure Julia (no GPU): the glue is the data-dependent middle of the
# two-stage object detectors. The reference ROIAlign values were captured from torchvision.ops.roi_align
# (torch 2.12) for both pixel conventions, so these pin the `aligned` flag to torchvision exactly.

const _G = ReactantServer.DetectionGlue

@testset "DetectionGlue" begin
    @testset "roi_align aligned flag vs torchvision" begin
        # feat: torch [1,8,8] with f[r,c] = r*8 + c (0-based). Julia feat[c=1, r+1, col+1].
        H = W = 8
        feat = Array{Float64}(undef, 1, H, W)
        for r in 0:H-1, c in 0:W-1
            feat[1, r+1, c+1] = r*8 + c
        end
        boxes = reshape(Float64[1.3, 2.1, 6.7, 5.9], 1, 4)
        scale, pooled, ratio = 0.5, 2, 2

        exp_aligned   = [9.025 10.375; 16.625 17.975]    # torch aligned=True  (half-pixel offset)
        exp_unaligned = [13.525 14.875; 21.125 22.475]   # torch aligned=False (torchvision detection)

        out = Array{Float64}(undef, 1, 1, pooled, pooled)
        _G.roi_align!(out, feat, boxes, scale; pooled=pooled, ratio=ratio, aligned=true)
        got_aligned = [out[1,1,ph,pw] for ph in 1:pooled, pw in 1:pooled]
        _G.roi_align!(out, feat, boxes, scale; pooled=pooled, ratio=ratio, aligned=false)
        got_unaligned = [out[1,1,ph,pw] for ph in 1:pooled, pw in 1:pooled]

        # wire-layout variant (pw,ph,C,K) must agree with roi_align!
        wire = Array{Float32}(undef, pooled, pooled, 1, 1)
        _G.roi_align_wire!(wire, feat, boxes, scale; pooled=pooled, ratio=ratio, aligned=false)
        got_wire = [Float64(wire[pw,ph,1,1]) for ph in 1:pooled, pw in 1:pooled]

        @test maximum(abs.(got_aligned   .- exp_aligned))   < 1e-6
        @test maximum(abs.(got_unaligned .- exp_unaligned)) < 1e-6
        @test maximum(abs.(got_wire      .- exp_unaligned)) < 1e-4
    end

    @testset "fast_rcnn_inference bg_first column selection" begin
        proposals = Float64[10 10 50 50; 20 20 60 60]
        deltas = zeros(Float64, 2, 3*4)        # zero deltas -> decoded == proposals
        # bg_first=true (torchvision): bg is col1; foreground cols 2,3 -> class ids 1,2.
        cls_first = Float64[0 5 0; 0 0 5]
        _,_,c1 = _G.fast_rcnn_inference(cls_first, deltas, proposals, 100, 100;
            score_thresh=0.05, nms_thresh=0.5, topk=100, bg_first=true)
        @test sort(c1) == [1, 2]
        # bg_first=false: bg is last col; foreground cols 1,2 -> class ids 0,1.
        cls_last = Float64[5 0 0; 0 5 0]
        _,_,c0 = _G.fast_rcnn_inference(cls_last, deltas, proposals, 100, 100;
            score_thresh=0.05, nms_thresh=0.5, topk=100, bg_first=false)
        @test sort(c0) == [0, 1]
    end
end
