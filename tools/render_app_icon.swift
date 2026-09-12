// Apply the macOS icon silhouette without repainting the supplied artwork.
import AppKit

guard CommandLine.arguments.count == 3 else {
    fatalError("Usage: render_app_icon.swift SOURCE.png OUTPUT.png")
}
let source = URL(fileURLWithPath: CommandLine.arguments[1])
let output = URL(fileURLWithPath: CommandLine.arguments[2])
guard let artwork = NSImage(contentsOf: source),
      let bitmap = NSBitmapImageRep(
        bitmapDataPlanes: nil, pixelsWide: 1024, pixelsHigh: 1024,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
        isPlanar: false, colorSpaceName: .deviceRGB,
        bytesPerRow: 0, bitsPerPixel: 0
      ), let context = NSGraphicsContext(bitmapImageRep: bitmap) else {
    fatalError("Cannot load artwork or allocate the icon canvas")
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
let outline = NSBezierPath(roundedRect: tile, xRadius: 185, yRadius: 185)
outline.addClip()
artwork.draw(in: tile, from: .zero, operation: .copy, fraction: 1)
NSGraphicsContext.restoreGraphicsState()

guard let png = bitmap.representation(using: .png, properties: [:]) else {
    fatalError("Cannot encode the transparent icon")
}
try png.write(to: output, options: .atomic)
