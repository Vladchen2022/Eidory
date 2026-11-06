// Apply the macOS icon silhouette without repainting the supplied artwork.
import AppKit
import SwiftUI

guard [3, 7].contains(CommandLine.arguments.count) else {
    fatalError("Usage: render_app_icon.swift SOURCE.png OUTPUT.png [CROP_X CROP_Y CROP_WIDTH CROP_HEIGHT]")
}
let source = URL(fileURLWithPath: CommandLine.arguments[1])
let output = URL(fileURLWithPath: CommandLine.arguments[2])
let sourceData = try Data(contentsOf: source)
guard let artwork = NSImage(data: sourceData),
      let sourceBitmap = NSBitmapImageRep(data: sourceData),
      let bitmap = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: 1024, pixelsHigh: 1024,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
        isPlanar: false, colorSpaceName: .deviceRGB,
        bytesPerRow: 0, bitsPerPixel: 0
      ), let context = NSGraphicsContext(bitmapImageRep: bitmap) else {
    fatalError("Cannot load artwork or allocate the icon canvas")
}

// Crop coordinates are source pixels, measured from the top-left. Resizing and
// clipping use the original bitmap, so the supplied mark is never redrawn.
artwork.size = NSSize(width: sourceBitmap.pixelsWide, height: sourceBitmap.pixelsHigh)
var sourceRect = NSRect(origin: .zero, size: artwork.size)
if CommandLine.arguments.count == 7 {
    let values = CommandLine.arguments[3...6].compactMap(Double.init)
    guard values.count == 4, values.allSatisfy({ $0.isFinite }),
          values[0] >= 0, values[1] >= 0, values[2] > 0,
          values[2] == values[3],
          values[0] + values[2] <= artwork.size.width,
          values[1] + values[3] <= artwork.size.height else {
        fatalError("Crop must be a square inside the source image")
    }
    sourceRect = NSRect(
        x: values[0], y: artwork.size.height - values[1] - values[3],
        width: values[2], height: values[3]
    )
}
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = context
context.imageInterpolation = .high
let canvas = NSRect(x: 0, y: 0, width: 1024, height: 1024)
NSColor.clear.setFill()
canvas.fill(using: .copy)

// The 824-point body gives legacy ICNS icons the same optical size as Dock tiles.
// Transparency must live in the asset; legacy ICNS assets are not rounded for us.
let tile = NSRect(x: 100, y: 100, width: 824, height: 824)
let outline = RoundedRectangle(cornerRadius: 185, style: .continuous).path(in: tile)
context.cgContext.addPath(outline.cgPath)
context.cgContext.clip()
artwork.draw(in: tile, from: sourceRect, operation: .copy, fraction: 1)
NSGraphicsContext.restoreGraphicsState()

guard let png = bitmap.representation(using: .png, properties: [:]) else {
    fatalError("Cannot encode the transparent icon")
}
try png.write(to: output, options: .atomic)
