import AppKit
import Darwin
import Foundation

private struct OverlayState: Codable {
    var imagePath: String? = nil
    var imageRevision: Int? = nil
    var opacity: Double? = nil
    var alwaysOnTop: Bool? = nil
    var clickThrough: Bool? = nil
    var adjustableFrame: Bool? = nil
    var showFrame: Bool? = nil
    var closeRequested: Bool? = nil
}

private final class OverlayCanvasView: NSView {
    var image: NSImage? {
        didSet { needsDisplay = true }
    }

    var contentOpacity: CGFloat = 1.0 {
        didSet { needsDisplay = true }
    }

    var showFrame: Bool = true {
        didSet { needsDisplay = true }
    }

    override var isOpaque: Bool { false }
    override var mouseDownCanMoveWindow: Bool { true }

    override func draw(_ dirtyRect: NSRect) {
        NSColor.clear.setFill()
        dirtyRect.fill()

        if let image {
            drawImage(image)
        }

        if showFrame {
            drawFrame()
        }
    }

    private func drawImage(_ image: NSImage) {
        let imageSize = image.size
        guard imageSize.width > 0, imageSize.height > 0 else { return }

        let available = bounds.insetBy(dx: 10, dy: 10)
        guard available.width > 0, available.height > 0 else { return }

        let scale = min(available.width / imageSize.width, available.height / imageSize.height)
        let width = imageSize.width * scale
        let height = imageSize.height * scale
        let target = NSRect(
            x: available.midX - width / 2,
            y: available.midY - height / 2,
            width: width,
            height: height
        )

        image.draw(
            in: target,
            from: NSRect(origin: .zero, size: imageSize),
            operation: .sourceOver,
            fraction: contentOpacity,
            respectFlipped: true,
            hints: [.interpolation: NSImageInterpolation.high]
        )
    }

    private func drawFrame() {
        let shadowRect = bounds.insetBy(dx: 6, dy: 6)
        let borderRect = bounds.insetBy(dx: 7, dy: 7)

        NSColor.black.withAlphaComponent(0.18).setStroke()
        let shadowPath = NSBezierPath(roundedRect: shadowRect, xRadius: 12, yRadius: 12)
        shadowPath.lineWidth = 4
        shadowPath.stroke()

        NSColor.white.withAlphaComponent(0.35).setStroke()
        let borderPath = NSBezierPath(roundedRect: borderRect, xRadius: 11, yRadius: 11)
        borderPath.lineWidth = 2
        borderPath.stroke()
    }
}

private enum OverlayControlAction {
    case toggleAlwaysOnTop
    case toggleClickThrough
    case toggleFrame
    case close
}

private final class OverlayControlsView: NSView {
    var alwaysOnTop: Bool = true {
        didSet { needsDisplay = true }
    }

    var clickThrough: Bool = false {
        didSet { needsDisplay = true }
    }

    var showFrame: Bool = true {
        didSet { needsDisplay = true }
    }

    var onAction: ((OverlayControlAction) -> Void)?

    private let buttonSize: CGFloat = 22
    private let buttonGap: CGFloat = 6

    override var isOpaque: Bool { false }

    override func draw(_ dirtyRect: NSRect) {
        NSColor.clear.setFill()
        dirtyRect.fill()

        for button in buttons() {
            drawButton(action: button.action, label: button.label, rect: button.rect, active: button.active)
        }
    }

    override func mouseDown(with event: NSEvent) {
        let point = convert(event.locationInWindow, from: nil)
        for button in buttons() where button.rect.contains(point) {
            onAction?(button.action)
            return
        }
    }

    private func buttons() -> [(action: OverlayControlAction, label: String, rect: NSRect, active: Bool)] {
        let y = (bounds.height - buttonSize) / 2
        let definitions: [(OverlayControlAction, String, Bool)] = [
            (.toggleAlwaysOnTop, "置", alwaysOnTop),
            (.toggleClickThrough, "穿", clickThrough),
            (.toggleFrame, "框", !showFrame),
            (.close, "×", false),
        ]
        return definitions.enumerated().map { index, definition in
            let x = CGFloat(index) * (buttonSize + buttonGap)
            return (
                action: definition.0,
                label: definition.1,
                rect: NSRect(x: x, y: y, width: buttonSize, height: buttonSize),
                active: definition.2
            )
        }
    }

    private func drawButton(action: OverlayControlAction, label: String, rect: NSRect, active: Bool) {
        let buttonRect = rect.insetBy(dx: 1, dy: 1)
        let path = NSBezierPath(roundedRect: buttonRect, xRadius: 10, yRadius: 10)
        let fillColor: NSColor
        if action == .close {
            fillColor = NSColor.black.withAlphaComponent(0.58)
        } else if active {
            fillColor = NSColor(calibratedRed: 0.12, green: 0.48, blue: 0.96, alpha: 0.86)
        } else {
            fillColor = NSColor.black.withAlphaComponent(0.52)
        }
        fillColor.setFill()
        path.fill()
        NSColor.white.withAlphaComponent(active ? 0.78 : 0.55).setStroke()
        path.lineWidth = 1
        path.stroke()

        let paragraph = NSMutableParagraphStyle()
        paragraph.alignment = .center
        let attributes: [NSAttributedString.Key: Any] = [
            .font: NSFont.systemFont(ofSize: action == .close ? 14 : 12, weight: .semibold),
            .foregroundColor: NSColor.white.withAlphaComponent(0.92),
            .paragraphStyle: paragraph,
        ]
        let textRect = NSRect(x: rect.minX, y: rect.minY + 3, width: rect.width, height: rect.height - 5)
        (label as NSString).draw(in: textRect, withAttributes: attributes)
    }
}

private final class OverlayController {
    private let stateURL: URL
    private let parentPID: pid_t
    private let view = OverlayCanvasView(frame: NSRect(x: 0, y: 0, width: 520, height: 520))
    private let controlsView = OverlayControlsView(frame: NSRect(origin: .zero, size: OverlayController.controlsSize))
    private let window: NSWindow
    private let controlsWindow: NSWindow
    private var timer: Timer?
    private var lastStateData: Data?
    private var lastImagePath: String?
    private var lastImageRevision: Int?
    private var currentState = OverlayState()

    private static let controlsSize = NSSize(width: 106, height: 22)

    init(stateURL: URL, parentPID: pid_t) {
        self.stateURL = stateURL
        self.parentPID = parentPID
        self.window = NSWindow(
            contentRect: OverlayController.initialFrame(),
            styleMask: [.borderless, .resizable],
            backing: .buffered,
            defer: false
        )
        self.controlsWindow = NSWindow(
            contentRect: NSRect(origin: .zero, size: OverlayController.controlsSize),
            styleMask: [.borderless],
            backing: .buffered,
            defer: false
        )
        configureWindow()
        configureControlsWindow()
    }

    func start() {
        applyStateIfNeeded(force: true)
        window.makeKeyAndOrderFront(nil)
        window.orderFrontRegardless()
        controlsWindow.orderFrontRegardless()
        positionControls()
        timer = Timer.scheduledTimer(withTimeInterval: 0.12, repeats: true) { [weak self] _ in
            self?.tick()
        }
        RunLoop.main.add(timer!, forMode: .common)
    }

    private func tick() {
        if parentPID > 0 && kill(parentPID, 0) != 0 {
            NSApp.terminate(nil)
            return
        }
        applyStateIfNeeded(force: false)
        positionControls()
    }

    private func configureWindow() {
        view.autoresizingMask = [.width, .height]
        window.contentView = view
        window.minSize = NSSize(width: 180, height: 140)
        window.isOpaque = false
        window.backgroundColor = .clear
        window.hasShadow = true
        window.titleVisibility = .hidden
        window.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        window.level = .floating
        window.isMovableByWindowBackground = true
        window.acceptsMouseMovedEvents = true
    }

    private func configureControlsWindow() {
        controlsView.autoresizingMask = [.width, .height]
        controlsView.onAction = { [weak self] action in
            self?.handleControlAction(action)
        }
        controlsWindow.contentView = controlsView
        controlsWindow.isOpaque = false
        controlsWindow.backgroundColor = .clear
        controlsWindow.hasShadow = false
        controlsWindow.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        controlsWindow.level = .floating
        controlsWindow.ignoresMouseEvents = false
    }

    private func applyStateIfNeeded(force: Bool) {
        guard let data = try? Data(contentsOf: stateURL) else { return }
        if !force && data == lastStateData { return }
        lastStateData = data

        guard let state = try? JSONDecoder().decode(OverlayState.self, from: data) else { return }
        currentState = state
        if state.closeRequested == true {
            NSApp.terminate(nil)
            return
        }

        let opacity = CGFloat(min(max(state.opacity ?? 1.0, 0.05), 1.0))
        view.contentOpacity = opacity
        view.showFrame = state.showFrame ?? true

        let adjustable = state.adjustableFrame ?? true
        var styleMask: NSWindow.StyleMask = [.borderless]
        if adjustable {
            styleMask.insert(.resizable)
        }
        if window.styleMask != styleMask {
            window.styleMask = styleMask
        }

        let clickThrough = state.clickThrough ?? false
        window.ignoresMouseEvents = clickThrough
        window.isMovableByWindowBackground = adjustable && !clickThrough
        window.level = (state.alwaysOnTop ?? true) ? .floating : .normal
        controlsWindow.level = window.level
        controlsView.alwaysOnTop = state.alwaysOnTop ?? true
        controlsView.clickThrough = clickThrough
        controlsView.showFrame = state.showFrame ?? true

        if lastImagePath != state.imagePath || lastImageRevision != state.imageRevision {
            lastImagePath = state.imagePath
            lastImageRevision = state.imageRevision
            if let imagePath = state.imagePath {
                view.image = NSImage(contentsOfFile: imagePath)
            } else {
                view.image = nil
            }
        }
        controlsWindow.orderFrontRegardless()
        positionControls()
    }

    private func handleControlAction(_ action: OverlayControlAction) {
        var state = currentState
        switch action {
        case .toggleAlwaysOnTop:
            state.alwaysOnTop = !(state.alwaysOnTop ?? true)
        case .toggleClickThrough:
            state.clickThrough = !(state.clickThrough ?? false)
        case .toggleFrame:
            state.showFrame = !(state.showFrame ?? true)
        case .close:
            NSApp.terminate(nil)
            return
        }
        writeState(state)
        applyStateIfNeeded(force: true)
    }

    private func writeState(_ state: OverlayState) {
        guard let data = try? JSONEncoder().encode(state) else { return }
        try? data.write(to: stateURL, options: .atomic)
    }

    private func positionControls() {
        let frame = window.frame
        let size = controlsWindow.frame.size
        let margin: CGFloat = 12
        let target = NSRect(
            x: frame.maxX - size.width - margin,
            y: frame.maxY - size.height - margin,
            width: size.width,
            height: size.height
        )
        controlsWindow.setFrame(target, display: true)
        controlsWindow.order(.above, relativeTo: window.windowNumber)
    }

    private static func initialFrame() -> NSRect {
        if let screen = NSScreen.main {
            let visible = screen.visibleFrame
            let width: CGFloat = 560
            let height: CGFloat = 560
            return NSRect(
                x: visible.midX - width / 2,
                y: visible.midY - height / 2,
                width: width,
                height: height
            )
        }
        return NSRect(x: 200, y: 200, width: 560, height: 560)
    }
}

private final class AppDelegate: NSObject, NSApplicationDelegate {
    private let controller: OverlayController

    init(controller: OverlayController) {
        self.controller = controller
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        controller.start()
    }
}

private func argumentValue(_ name: String) -> String? {
    let args = CommandLine.arguments
    guard let index = args.firstIndex(of: name), args.indices.contains(index + 1) else {
        return nil
    }
    return args[index + 1]
}

guard let statePath = argumentValue("--state") else {
    fputs("missing --state\\n", stderr)
    exit(2)
}

private let parentPID = pid_t(Int32(argumentValue("--parent-pid") ?? "0") ?? 0)
private let controller = OverlayController(stateURL: URL(fileURLWithPath: statePath), parentPID: parentPID)
private let delegate = AppDelegate(controller: controller)
private let app = NSApplication.shared
app.setActivationPolicy(.accessory)
app.delegate = delegate
app.run()
