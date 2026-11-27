#!/usr/bin/env swift
// Set the default input/output audio device by name via CoreAudio.
// Usage: audio_route list | audio_route input <name> | audio_route output <name>
// Requires the devices to be installed (e.g. BlackHole 2ch).

import CoreAudio
import Foundation

func deviceIDs() -> [AudioObjectID] {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var size: UInt32 = 0
    AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size)
    let count = Int(size) / MemoryLayout<AudioObjectID>.size
    var ids = [AudioObjectID](repeating: 0, count: count)
    AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &ids)
    return ids
}

func deviceName(_ id: AudioObjectID) -> String {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioObjectPropertyName,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var name: CFString = "" as CFString
    var size = UInt32(MemoryLayout<CFString>.size)
    AudioObjectGetPropertyData(id, &address, 0, nil, &size, &name)
    return name as String
}

func findDevice(_ wanted: String) -> AudioObjectID? {
    for id in deviceIDs() where deviceName(id).localizedCaseInsensitiveContains(wanted) {
        return id
    }
    return nil
}

func setDefault(_ selector: AudioObjectPropertySelector, _ id: AudioObjectID) -> Bool {
    var address = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var value = id
    let size = UInt32(MemoryLayout<AudioObjectID>.size)
    return AudioObjectSetPropertyData(
        AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, size, &value
    ) == noErr
}

let args = CommandLine.arguments
guard args.count >= 2 else {
    print("usage: audio_route list | input <name> | output <name>")
    exit(2)
}
switch args[1] {
case "list":
    for id in deviceIDs() { print(deviceName(id)) }
case "input", "output":
    guard args.count >= 3, let id = findDevice(args[2]) else {
        print("device not found: \(args.dropFirst(2).joined(separator: " "))")
        exit(3)
    }
    let selector = args[1] == "input"
        ? kAudioHardwarePropertyDefaultInputDevice
        : kAudioHardwarePropertyDefaultOutputDevice
    print(setDefault(selector, id) ? "set \(args[1]) to \(deviceName(id))" : "failed")
default:
    exit(2)
}
