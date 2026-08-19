#!/usr/bin/env swift

import AVFoundation
import Foundation
@preconcurrency import Speech

struct Options {
    var locale = "ja-JP"
    var durationSeconds = 600
    var silenceMilliseconds = 500
    var speechDecibels: Float = -38
    var allowNetworkRecognition = false
}

func parseOptions() -> Options {
    var options = Options()
    var arguments = Array(CommandLine.arguments.dropFirst())
    while !arguments.isEmpty {
        let flag = arguments.removeFirst()
        func value() -> String {
            guard !arguments.isEmpty else {
                FileHandle.standardError.write(Data("missing value for \(flag)\n".utf8))
                exit(2)
            }
            return arguments.removeFirst()
        }
        switch flag {
        case "--locale": options.locale = value()
        case "--duration": options.durationSeconds = Int(value()) ?? 0
        case "--silence-ms": options.silenceMilliseconds = Int(value()) ?? 0
        case "--speech-db": options.speechDecibels = Float(value()) ?? 0
        case "--allow-network-recognition": options.allowNetworkRecognition = true
        case "--help":
            print("usage: macos-speech-events [--locale ja-JP] [--duration 600] [--silence-ms 500] [--speech-db -38] [--allow-network-recognition]")
            exit(0)
        default:
            FileHandle.standardError.write(Data("unknown argument: \(flag)\n".utf8))
            exit(2)
        }
    }
    guard options.durationSeconds > 0, options.silenceMilliseconds >= 100 else {
        FileHandle.standardError.write(Data("duration must be positive and silence-ms at least 100\n".utf8))
        exit(2)
    }
    return options
}

final class EventEmitter: @unchecked Sendable {
    private let started = DispatchTime.now().uptimeNanoseconds
    private let lock = NSLock()

    func milliseconds() -> Int {
        Int((DispatchTime.now().uptimeNanoseconds - started) / 1_000_000)
    }

    func emit(_ type: String, text: String? = nil) {
        var object: [String: Any] = ["type": type, "at_ms": milliseconds()]
        if let text, !text.isEmpty { object["text"] = text }
        guard let data = try? JSONSerialization.data(withJSONObject: object),
              let line = String(data: data, encoding: .utf8) else { return }
        lock.lock()
        print(line)
        fflush(stdout)
        lock.unlock()
    }
}

final class RecognitionStream: @unchecked Sendable {
    private let recognizer: SFSpeechRecognizer
    private let onDevice: Bool
    private let lock = NSLock()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    private var generation = 0

    init(recognizer: SFSpeechRecognizer, onDevice: Bool) {
        self.recognizer = recognizer
        self.onDevice = onDevice
    }

    func start(handler: @escaping @Sendable (SFSpeechRecognitionResult?, Error?) -> Void) {
        lock.lock()
        generation += 1
        let currentGeneration = generation
        let next = SFSpeechAudioBufferRecognitionRequest()
        next.shouldReportPartialResults = true
        next.requiresOnDeviceRecognition = onDevice
        next.taskHint = .dictation
        request = next
        task = recognizer.recognitionTask(with: next) { [weak self] result, error in
            guard let self else { return }
            self.lock.lock()
            let isCurrent = currentGeneration == self.generation
            self.lock.unlock()
            if isCurrent { handler(result, error) }
        }
        lock.unlock()
    }

    func append(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        request?.append(buffer)
        lock.unlock()
    }

    func restart(handler: @escaping @Sendable (SFSpeechRecognitionResult?, Error?) -> Void) {
        lock.lock()
        let oldRequest = request
        let oldTask = task
        request = nil
        task = nil
        lock.unlock()
        oldRequest?.endAudio()
        oldTask?.finish()
        start(handler: handler)
    }

    func finish() {
        lock.lock()
        generation += 1
        let finalRequest = request
        let finalTask = task
        request = nil
        task = nil
        lock.unlock()
        finalRequest?.endAudio()
        finalTask?.finish()
    }
}

let options = parseOptions()
let emitter = EventEmitter()
let authorization = DispatchSemaphore(value: 0)
var authorized = false
SFSpeechRecognizer.requestAuthorization { status in
    authorized = status == .authorized
    authorization.signal()
}
authorization.wait()
guard authorized else {
    FileHandle.standardError.write(Data("speech recognition permission was not granted\n".utf8))
    exit(3)
}

guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: options.locale)), recognizer.isAvailable else {
    FileHandle.standardError.write(Data("speech recognizer is unavailable for \(options.locale)\n".utf8))
    exit(4)
}
if !options.allowNetworkRecognition && !recognizer.supportsOnDeviceRecognition {
    FileHandle.standardError.write(Data("on-device recognition is unavailable; pass --allow-network-recognition only with informed consent\n".utf8))
    exit(4)
}

let engine = AVAudioEngine()
let input = engine.inputNode
let format = input.outputFormat(forBus: 0)
guard format.sampleRate > 0, format.channelCount > 0 else {
    FileHandle.standardError.write(Data("microphone input is unavailable\n".utf8))
    exit(5)
}

let stateQueue = DispatchQueue(label: "cueloop.macos-speech.state")
var speaking = false
var lastSpeechAt = 0
var lastTranscript = ""
var committedTranscript = ""
var recognitionStartedAt = 0
var participantRequestedStop = false
var interventionsPaused = false

func currentTurnText() -> String {
    guard lastTranscript.hasPrefix(committedTranscript) else { return lastTranscript }
    return String(lastTranscript.dropFirst(committedTranscript.count))
        .trimmingCharacters(in: .whitespacesAndNewlines)
}

let stream = RecognitionStream(recognizer: recognizer, onDevice: !options.allowNetworkRecognition)
let recognitionHandler: @Sendable (SFSpeechRecognitionResult?, Error?) -> Void = { result, error in
    stateQueue.async {
        if let result {
            let transcript = result.bestTranscription.formattedString.trimmingCharacters(in: .whitespacesAndNewlines)
            if !transcript.isEmpty && transcript != lastTranscript {
                lastTranscript = transcript
                let turnText = currentTurnText()
                if !turnText.isEmpty {
                    emitter.emit("transcript_partial", text: turnText)
                }
            }
        }
        if let error {
            emitter.emit("adapter_warning", text: error.localizedDescription)
            FileHandle.standardError.write(Data("recognition warning: \(error.localizedDescription)\n".utf8))
        }
    }
}
stream.start(handler: recognitionHandler)

input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
    stream.append(buffer)
    guard let samples = buffer.floatChannelData?[0] else { return }
    let count = Int(buffer.frameLength)
    guard count > 0 else { return }
    var sum: Float = 0
    for index in 0..<count { sum += samples[index] * samples[index] }
    let rms = sqrt(sum / Float(count))
    let decibels = 20 * log10(max(rms, 0.000_001))
    if decibels >= options.speechDecibels {
        stateQueue.async {
            lastSpeechAt = emitter.milliseconds()
            if !speaking {
                speaking = true
                emitter.emit("speech_started")
            }
        }
    }
}

do {
    engine.prepare()
    try engine.start()
} catch {
    FileHandle.standardError.write(Data("failed to start microphone: \(error.localizedDescription)\n".utf8))
    exit(5)
}

FileHandle.standardError.write(Data("recording active; raw audio is not retained; press Ctrl-C to stop\n".utf8))
let timer = DispatchSource.makeTimerSource(queue: stateQueue)
timer.schedule(deadline: .now() + .milliseconds(100), repeating: .milliseconds(100))
timer.setEventHandler {
    let now = emitter.milliseconds()
    if speaking && now - lastSpeechAt >= options.silenceMilliseconds {
        speaking = false
        let turnText = currentTurnText()
        if !turnText.isEmpty {
            emitter.emit("transcript_final", text: turnText)
            committedTranscript = lastTranscript
        }
        emitter.emit("speech_ended")
    }
    if !speaking && now - recognitionStartedAt >= 45_000 && now - lastSpeechAt >= 3_000 {
        stream.restart(handler: recognitionHandler)
        recognitionStartedAt = now
        lastTranscript = ""
        committedTranscript = ""
        FileHandle.standardError.write(Data("speech recognition task rolled over\n".utf8))
    }
    emitter.emit("tick")
}
timer.resume()

signal(SIGINT, SIG_IGN)
let interrupt = DispatchSource.makeSignalSource(signal: SIGINT, queue: stateQueue)
interrupt.setEventHandler {
    participantRequestedStop = true
    CFRunLoopStop(CFRunLoopGetMain())
}
interrupt.resume()
signal(SIGUSR1, SIG_IGN)
let interventionToggle = DispatchSource.makeSignalSource(signal: SIGUSR1, queue: stateQueue)
interventionToggle.setEventHandler {
    interventionsPaused.toggle()
    emitter.emit(interventionsPaused ? "interventions_paused" : "interventions_resumed")
    FileHandle.standardError.write(
        Data((interventionsPaused ? "interventions paused\n" : "interventions resumed\n").utf8)
    )
}
interventionToggle.resume()
FileHandle.standardError.write(
    Data("process id: \(ProcessInfo.processInfo.processIdentifier); send SIGUSR1 to toggle interventions\n".utf8)
)
DispatchQueue.main.asyncAfter(deadline: .now() + .seconds(options.durationSeconds)) {
    CFRunLoopStop(CFRunLoopGetMain())
}
CFRunLoopRun()

timer.cancel()
engine.stop()
input.removeTap(onBus: 0)
stateQueue.sync {
    if speaking {
        let turnText = currentTurnText()
        if !turnText.isEmpty { emitter.emit("transcript_final", text: turnText) }
        emitter.emit("speech_ended")
        speaking = false
    }
}
// Give the core's endpoint detector a real silent interval so the final turn
// is committed on duration expiry or Ctrl-C without speaking after shutdown.
Thread.sleep(forTimeInterval: 0.7)
stateQueue.sync {
    emitter.emit("tick")
    if participantRequestedStop { emitter.emit("participant_stop") }
    emitter.emit("session_ended")
}
stream.finish()
