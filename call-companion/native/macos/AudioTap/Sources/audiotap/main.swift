// audiotap: streams the microphone ("me") and system audio ("them") to stdout as 16 kHz mono Int16 frames.
//
// Frame: 1 byte channel (0 = me, 1 = them), UInt32 little-endian byte count, then that many PCM bytes.
// Diagnostics go to stderr. Exit codes: 0 clean stop, 2 microphone, 3 system audio (usually permission), 4 stream died.
//
// Build:  swift build -c release   (macOS 13+, Xcode command line tools)
// Needs:  Microphone and Screen & System Audio Recording permission for the app that launches it (Terminal, iTerm).

import AVFoundation
import CoreMedia
import Foundation
import ScreenCaptureKit

let targetRate = 16_000.0

func log(_ message: String) {
    FileHandle.standardError.write(Data(("audiotap: " + message + "\n").utf8))
}

func fail(_ message: String, code: Int32) -> Never {
    log(message)
    exit(code)
}

enum TapError: Error, CustomStringConvertible {
    case noMicrophone
    case noDisplay

    var description: String {
        switch self {
        case .noMicrophone: return "нет входного аудиоустройства"
        case .noDisplay: return "нет дисплея для захвата звука"
        }
    }
}

final class FrameWriter {
    private let out = FileHandle.standardOutput
    private let lock = NSLock()

    func write(channel: UInt8, samples: UnsafeBufferPointer<Int16>) {
        guard let base = samples.baseAddress, samples.count > 0 else { return }
        let byteCount = samples.count * MemoryLayout<Int16>.size
        var length = UInt32(byteCount).littleEndian
        var frame = Data(capacity: 5 + byteCount)
        frame.append(channel)
        withUnsafeBytes(of: &length) { frame.append(contentsOf: $0) }
        frame.append(Data(bytes: base, count: byteCount))
        lock.lock()
        defer { lock.unlock() }
        do { try out.write(contentsOf: frame) } catch { exit(0) }  // the reader went away: stop quietly
    }
}

/// Converts whatever the source delivers to 16 kHz mono Int16 and writes it as frames of one channel.
final class Resampler {
    private let target: AVAudioFormat
    private var converter: AVAudioConverter?
    private var sourceFormat: AVAudioFormat?
    private let channel: UInt8
    private let writer: FrameWriter

    init(channel: UInt8, writer: FrameWriter) {
        self.channel = channel
        self.writer = writer
        self.target = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: targetRate, channels: 1, interleaved: true)!
    }

    func feed(_ buffer: AVAudioPCMBuffer) {
        if converter == nil || sourceFormat != buffer.format {
            sourceFormat = buffer.format
            converter = AVAudioConverter(from: buffer.format, to: target)
        }
        guard let converter else { return }
        let ratio = targetRate / buffer.format.sampleRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 32
        guard let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { return }
        var supplied = false
        var error: NSError?
        _ = converter.convert(to: output, error: &error) { _, status in
            if supplied {
                status.pointee = .noDataNow
                return nil
            }
            supplied = true
            status.pointee = .haveData
            return buffer
        }
        guard error == nil, output.frameLength > 0, let data = output.int16ChannelData else { return }
        writer.write(channel: channel, samples: UnsafeBufferPointer(start: data[0], count: Int(output.frameLength)))
    }
}

final class Microphone {
    private let engine = AVAudioEngine()
    private let resampler: Resampler

    init(writer: FrameWriter) {
        resampler = Resampler(channel: 0, writer: writer)
    }

    func start() throws {
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0 else { throw TapError.noMicrophone }
        input.installTap(onBus: 0, bufferSize: 4096, format: format) { [resampler] buffer, _ in
            resampler.feed(buffer)
        }
        engine.prepare()
        try engine.start()
    }
}

final class SystemAudio: NSObject, SCStreamOutput, SCStreamDelegate {
    private var stream: SCStream?
    private let queue = DispatchQueue(label: "audiotap.system")
    private let resampler: Resampler

    init(writer: FrameWriter) {
        resampler = Resampler(channel: 1, writer: writer)
    }

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else { throw TapError.noDisplay }
        let filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.excludesCurrentProcessAudio = true
        config.sampleRate = 48_000
        config.channelCount = 2
        config.width = 2  // video is not wanted; the smallest possible frame keeps the cost negligible
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue)
        try await stream.startCapture()
        self.stream = stream
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid, let buffer = Self.pcm(from: sampleBuffer) else { return }
        resampler.feed(buffer)
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        fail("поток системного звука остановлен: \(error.localizedDescription)", code: 4)
    }

    private static func pcm(from sampleBuffer: CMSampleBuffer) -> AVAudioPCMBuffer? {
        guard let description = CMSampleBufferGetFormatDescription(sampleBuffer),
              let asbd = CMAudioFormatDescriptionGetStreamBasicDescription(description),
              let format = AVAudioFormat(streamDescription: asbd)
        else { return nil }
        let frames = AVAudioFrameCount(CMSampleBufferGetNumSamples(sampleBuffer))
        guard frames > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else { return nil }
        buffer.frameLength = frames
        let status = CMSampleBufferCopyPCMDataIntoAudioBufferList(
            sampleBuffer, at: 0, frameCount: Int32(frames), into: buffer.mutableAudioBufferList)
        return status == noErr ? buffer : nil
    }
}

func microphoneAllowed() -> Bool {
    switch AVCaptureDevice.authorizationStatus(for: .audio) {
    case .authorized:
        return true
    case .notDetermined:
        let semaphore = DispatchSemaphore(value: 0)
        var granted = false
        AVCaptureDevice.requestAccess(for: .audio) {
            granted = $0
            semaphore.signal()
        }
        semaphore.wait()
        return granted
    default:
        return false
    }
}

guard microphoneAllowed() else {
    fail("нет доступа к микрофону: System Settings → Privacy & Security → Microphone", code: 2)
}

let writer = FrameWriter()
let microphone = Microphone(writer: writer)
let system = SystemAudio(writer: writer)

signal(SIGTERM, SIG_IGN)
signal(SIGINT, SIG_IGN)
let stopSources = [SIGTERM, SIGINT].map { number -> DispatchSourceSignal in
    let source = DispatchSource.makeSignalSource(signal: number, queue: .main)
    source.setEventHandler { exit(0) }
    source.resume()
    return source
}

do { try microphone.start() } catch { fail("микрофон: \(error)", code: 2) }
do {
    try await system.start()
} catch {
    fail("системный звук: \(error). Разрешите «Запись экрана и системного звука» приложению, из которого запущен audiotap.", code: 3)
}
log("слушаю")
dispatchMain()
