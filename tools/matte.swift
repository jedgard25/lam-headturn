// matte <in> <out.png>: Apple Vision subject lift -> RGBA PNG (alpha = foreground mask)
import Foundation
import Vision
import CoreImage
import ImageIO

let a = CommandLine.arguments
guard a.count == 3 else { FileHandle.standardError.write("usage: matte in out.png\n".data(using: .utf8)!); exit(2) }
let url = URL(fileURLWithPath: a[1])
guard let src = CGImageSourceCreateWithURL(url as CFURL, nil), let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { print("cannot read image"); exit(1) }
let req = VNGenerateForegroundInstanceMaskRequest()
let h = VNImageRequestHandler(cgImage: cg, options: [:])
try h.perform([req])
guard let obs = req.results?.first, !obs.allInstances.isEmpty else { print("no subject found"); exit(3) }
let maskBuf = try obs.generateScaledMaskForImage(forInstances: obs.allInstances, from: h)
let ci = CIImage(cgImage: cg)
let mask = CIImage(cvPixelBuffer: maskBuf)
let clear = CIImage(color: .clear).cropped(to: ci.extent)
let out = ci.applyingFilter("CIBlendWithMask", parameters: [kCIInputBackgroundImageKey: clear, kCIInputMaskImageKey: mask])
let ctx = CIContext()
try ctx.writePNGRepresentation(of: out, to: URL(fileURLWithPath: a[2]), format: CIFormat.RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
