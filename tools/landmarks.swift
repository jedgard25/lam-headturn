// landmarks <in> : Apple Vision face landmarks -> JSON {w,h,yaw,roll,pitch,pts:{name:[[x,y],...]}} in pixel coords (y down)
import Foundation
import Vision
import ImageIO

let a = CommandLine.arguments
guard a.count == 2, let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: a[1]) as CFURL, nil),
      let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { print("{\"error\":\"cannot read image\"}"); exit(1) }
let W = Double(cg.width), H = Double(cg.height)
let req = VNDetectFaceLandmarksRequest()
req.revision = VNDetectFaceLandmarksRequestRevision3
try VNImageRequestHandler(cgImage: cg, options: [:]).perform([req])
guard let f = (req.results ?? []).max(by: { $0.boundingBox.width < $1.boundingBox.width }), let lm = f.landmarks else {
    print("{\"error\":\"no face\"}"); exit(3) }
let bb = f.boundingBox
func conv(_ r: VNFaceLandmarkRegion2D?) -> [[Double]] {
    guard let r = r else { return [] }
    return r.normalizedPoints.map { p in [(bb.minX + Double(p.x) * bb.width) * W, (1 - (bb.minY + Double(p.y) * bb.height)) * H] }
}
let regions: [(String, VNFaceLandmarkRegion2D?)] = [
    ("contour", lm.faceContour), ("leftEye", lm.leftEye), ("rightEye", lm.rightEye), ("leftBrow", lm.leftEyebrow),
    ("rightBrow", lm.rightEyebrow), ("nose", lm.nose), ("noseCrest", lm.noseCrest), ("medianLine", lm.medianLine),
    ("outerLips", lm.outerLips), ("innerLips", lm.innerLips), ("leftPupil", lm.leftPupil), ("rightPupil", lm.rightPupil)]
var pts: [String: [[Double]]] = [:]
for (n, r) in regions { pts[n] = conv(r) }
let o: [String: Any] = ["w": W, "h": H, "yaw": f.yaw?.doubleValue ?? 0, "roll": f.roll?.doubleValue ?? 0,
                        "pitch": f.pitch?.doubleValue ?? 0, "pts": pts]
print(String(data: try JSONSerialization.data(withJSONObject: o), encoding: .utf8)!)
