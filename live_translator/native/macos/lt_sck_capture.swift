// Live Translator — ScreenCaptureKit audio capture helper (macOS 13+)
//
// Adapted from the system-audio helper in jt-live-whisper; adds *per-application* capture.
// Only audio is consumed: no screen output is attached, so no video frames ever flow.
//
// Build (done automatically on first use by live_translator/audio/macos.py):
//   swiftc -O -o lt-sck-audio lt_sck_capture.swift \
//       -framework ScreenCaptureKit -framework AVFoundation -framework CoreMedia -framework CoreGraphics
//
// Usage:
//   lt-sck-audio --check                 -> one JSON line {available, permission, macos}
//   lt-sck-audio --request               -> trigger the "Screen Recording" permission prompt
//   lt-sck-audio --list-apps             -> one JSON line {"apps":[{bundle,name,pid}, ...]}
//   lt-sck-audio [--app BUNDLE_ID] [--rate 48000] [--channels 2]
//                                        -> raw interleaved float32 PCM on stdout (logs on stderr)
//                                           without --app: all system audio
import AVFoundation
import CoreGraphics
import CoreMedia
import Foundation
import ScreenCaptureKit

func writeErr(_ s: String) {
    FileHandle.standardError.write((s + "\n").data(using: .utf8)!)
}

func emitJSON(_ dict: [String: Any]) {
    let data = try! JSONSerialization.data(withJSONObject: dict, options: [.sortedKeys])
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
}

func supportsSCK() -> Bool {
    if #available(macOS 13.0, *) { return true }
    return false
}

// ── argument parsing ────────────────────────────────────────
var mode = "stream"
var rate = 48000
var channels = 2
var appBundle: String? = nil

var argIdx = 1
let argv = CommandLine.arguments
while argIdx < argv.count {
    switch argv[argIdx] {
    case "--check": mode = "check"
    case "--request": mode = "request"
    case "--list-apps": mode = "list"
    case "--app":
        argIdx += 1
        if argIdx < argv.count { appBundle = argv[argIdx] }
    case "--rate":
        argIdx += 1
        if argIdx < argv.count, let v = Int(argv[argIdx]) { rate = v }
    case "--channels":
        argIdx += 1
        if argIdx < argv.count, let v = Int(argv[argIdx]) { channels = max(1, min(2, v)) }
    default:
        writeErr("[sck] unknown argument: \(argv[argIdx])")
    }
    argIdx += 1
}

if mode == "check" {
    let v = ProcessInfo.processInfo.operatingSystemVersion
    emitJSON([
        "available": supportsSCK(),
        "permission": CGPreflightScreenCaptureAccess(),
        "macos": "\(v.majorVersion).\(v.minorVersion).\(v.patchVersion)",
    ])
    exit(0)
}

if mode == "request" {
    let granted = CGRequestScreenCaptureAccess()
    emitJSON(["permission": granted])
    exit(granted ? 0 : 1)
}

guard supportsSCK() else {
    writeErr("[sck] requires macOS 13.0 or later")
    exit(2)
}

// ── audio tap ───────────────────────────────────────────────
@available(macOS 13.0, *)
final class AudioTap: NSObject, SCStreamOutput, SCStreamDelegate {
    private let out = FileHandle.standardOutput
    private let targetChannels: Int
    private var scratch = [Float]()

    init(targetChannels: Int) {
        self.targetChannels = targetChannels
        super.init()
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer,
                of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid else { return }
        guard let fmt = sampleBuffer.formatDescription?.audioStreamBasicDescription else { return }
        let srcChannels = Int(fmt.mChannelsPerFrame)
        guard srcChannels > 0 else { return }

        try? sampleBuffer.withAudioBufferList { abl, _ in
            let buffers = abl.unsafePointer.pointee
            guard buffers.mNumberBuffers > 0 else { return }

            // ScreenCaptureKit delivers float32, usually non-interleaved (one buffer per channel).
            let isPlanar = (fmt.mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0
            var planes = [UnsafePointer<Float>]()
            var frameCount = 0
            for buf in abl {
                guard let data = buf.mData else { continue }
                planes.append(UnsafePointer(data.assumingMemoryBound(to: Float.self)))
                let perBufferChannels = isPlanar ? 1 : Int(buf.mNumberChannels)
                frameCount = max(frameCount, Int(buf.mDataByteSize) / 4 / max(1, perBufferChannels))
            }
            guard frameCount > 0, !planes.isEmpty else { return }

            let outCh = self.targetChannels
            if self.scratch.count != frameCount * outCh {
                self.scratch = [Float](repeating: 0, count: frameCount * outCh)
            }
            if isPlanar {
                for ch in 0..<outCh {
                    let src = planes[min(ch, planes.count - 1)]
                    for f in 0..<frameCount { self.scratch[f * outCh + ch] = src[f] }
                }
            } else {
                let src = planes[0]
                for f in 0..<frameCount {
                    for ch in 0..<outCh {
                        self.scratch[f * outCh + ch] = src[f * srcChannels + min(ch, srcChannels - 1)]
                    }
                }
            }
            self.scratch.withUnsafeBufferPointer { ptr in
                let data = Data(buffer: ptr)
                // EPIPE = the Python side closed the pipe → just exit.
                if write(self.out.fileDescriptor, (data as NSData).bytes, data.count) < 0 { exit(0) }
            }
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        writeErr("[sck] stream stopped: \(error.localizedDescription)")
        exit(3)
    }
}

// SCStream keeps its delegate weakly and a local stream would be released when the function
// returns → capture silently stops. Hold both for the process lifetime.
var gStream: AnyObject?
var gTap: AnyObject?

@available(macOS 13.0, *)
func fetchContent() async -> SCShareableContent {
    do {
        return try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
    } catch {
        writeErr("[sck] cannot query shareable content (Screen Recording permission?): \(error.localizedDescription)")
        exit(5)
    }
}

@available(macOS 13.0, *)
func listApps() async {
    let content = await fetchContent()
    let myPid = ProcessInfo.processInfo.processIdentifier
    let apps: [[String: Any]] = content.applications
        .filter { !$0.applicationName.isEmpty && !$0.bundleIdentifier.isEmpty && $0.processID != myPid }
        .map { ["bundle": $0.bundleIdentifier, "name": $0.applicationName, "pid": Int($0.processID)] }
    emitJSON(["apps": apps])
    exit(0)
}

@available(macOS 13.0, *)
func runCapture(rate: Int, channels: Int, appBundle: String?) async {
    guard CGPreflightScreenCaptureAccess() else {
        writeErr("[sck] Screen Recording permission not granted")
        exit(4)
    }
    let content = await fetchContent()
    guard let display = content.displays.first else {
        writeErr("[sck] no display found")
        exit(6)
    }

    let filter: SCContentFilter
    if let bundle = appBundle {
        let apps = content.applications.filter { $0.bundleIdentifier == bundle }
        guard !apps.isEmpty else {
            writeErr("[sck] application not running or has no capturable content: \(bundle)")
            exit(8)
        }
        filter = SCContentFilter(display: display, including: apps, exceptingWindows: [])
    } else {
        filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
    }

    let config = SCStreamConfiguration()
    config.capturesAudio = true
    config.excludesCurrentProcessAudio = true
    config.sampleRate = rate
    config.channelCount = channels
    config.width = 2                     // we never read video; keep it tiny
    config.height = 2
    config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
    config.queueDepth = 6

    let tap = AudioTap(targetChannels: channels)
    let stream = SCStream(filter: filter, configuration: config, delegate: tap)
    do {
        try stream.addStreamOutput(tap, type: .audio, sampleHandlerQueue: DispatchQueue(label: "lt.sck.audio"))
        try await stream.startCapture()
    } catch {
        writeErr("[sck] startCapture failed: \(error.localizedDescription)")
        exit(7)
    }
    gStream = stream
    gTap = tap
    writeErr("[sck] ready rate=\(rate) channels=\(channels) app=\(appBundle ?? "<system>")")
}

if #available(macOS 13.0, *) {
    signal(SIGINT) { _ in exit(0) }
    signal(SIGTERM) { _ in exit(0) }
    signal(SIGPIPE) { _ in exit(0) }
    if mode == "list" {
        Task { await listApps() }
    } else {
        Task { await runCapture(rate: rate, channels: channels, appBundle: appBundle) }
    }
    RunLoop.main.run()
}
