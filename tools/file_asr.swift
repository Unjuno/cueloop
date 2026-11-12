// One-shot file transcription with the same SFSpeechRecognizer engine the
// live macOS adapter uses. Used by scripts/evaluate_asr.py to score a local
// Japanese speech corpus against CUELOOP's production ASR path.
//
// Build: swiftc tools/file_asr.swift -o .build/lux-file-asr \
//   -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist \
//   -Xlinker tools/file_asr.Info.plist
// Usage: .build/lux-file-asr <path.wav>

import Foundation
import Speech

var stdoutData = Data()

guard CommandLine.arguments.count == 2 else {
    FileHandle.standardError.write(Data("usage: lux-file-asr <path.wav>\n".utf8))
    exit(2)
}

let path = CommandLine.arguments[1]
let url = URL(fileURLWithPath: path)

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

let locale = Locale(identifier: "ja-JP")
guard let recognizer = SFSpeechRecognizer(locale: locale), recognizer.isAvailable else {
    FileHandle.standardError.write(Data("recognizer is unavailable for ja-JP\n".utf8))
    exit(4)
}

let request = SFSpeechURLRecognitionRequest(url: url)
// On-device preferred; fall back to network recognition if the model is
// unavailable, mirroring the live adapter's behaviour on missing models.
request.requiresOnDeviceRecognition = false
request.taskHint = .dictation

let semaphore = DispatchSemaphore(value: 0)
var finalTranscript = ""
var finalConfidence = 0.0
var finalError: Error?

let task = recognizer.recognitionTask(with: request) { result, error in
    if let error, (error as NSError).domain == "k_final" {
        return
    }
    if let result {
        let transcription = result.bestTranscription.formattedString
        let segments = result.bestTranscription.segments
        let confidence = segments.isEmpty ? 1.0
            : segments.reduce(0.0, { $0 + $1.confidence }) / Double(segments.count)
        if result.isFinal {
            finalTranscript = transcription
            finalConfidence = confidence
            semaphore.signal()
        }
    }
    if let error, (error as NSError).code != 203, (error as NSError).domain != "216" {
        finalError = error
        semaphore.signal()
    }
}

_ = task
_ = semaphore.wait(timeout: .now() + 60)

let output: [String: Any]
if let error = finalError {
    output = [
        "wav": path,
        "error": error.localizedDescription,
    ]
} else {
    output = [
        "wav": path,
        "transcript": finalTranscript,
        "confidence": finalConfidence,
    ]
}

do {
    let data = try JSONSerialization.data(withJSONObject: output, options: [])
    stdoutData = data
} catch {
    FileHandle.standardError.write(Data("could not encode result: \(error.localizedDescription)\n".utf8))
    exit(2)
}

FileHandle.standardOutput.write(stdoutData)
FileHandle.standardOutput.write(Data("\n".utf8))
