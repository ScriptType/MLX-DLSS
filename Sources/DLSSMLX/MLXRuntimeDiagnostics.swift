import MLX
import os

public struct MLXMemorySnapshot: Equatable, Sendable {
    public let activeBytes: UInt64
    public let cacheBytes: UInt64
    public let peakActiveBytes: UInt64

    init(_ snapshot: Memory.Snapshot) {
        self.activeBytes = UInt64(max(0, snapshot.activeMemory))
        self.cacheBytes = UInt64(max(0, snapshot.cacheMemory))
        self.peakActiveBytes = UInt64(max(0, snapshot.peakMemory))
    }
}

public enum MLXRuntimeDiagnosticsError: Error, Equatable, Sendable {
    case cacheLimitTooLarge(UInt64)
}

public enum MLXRuntimeDiagnostics {
    /// Per-frame stage intervals for Instruments; nearly free while nothing records.
    public static let signposter = OSSignposter(subsystem: "com.scripttype.mlxdlss", category: .pointsOfInterest)

    public static func stage<T>(_ name: StaticString, isolation: isolated (any Actor)? = #isolation,
                                _ body: () async throws -> T) async rethrows -> T {
        let state = signposter.beginInterval(name)
        defer { signposter.endInterval(name, state) }
        return try await body()
    }

    public static func memorySnapshot() -> MLXMemorySnapshot {
        MLXMemorySnapshot(Memory.snapshot())
    }

    public static func resetPeakMemory() {
        Memory.peakMemory = 0
    }

    public static var cacheLimitBytes: UInt64 {
        UInt64(max(0, Memory.cacheLimit))
    }

    public static func setCacheLimitBytes(_ bytes: UInt64) throws {
        guard bytes <= UInt64(Int.max) else {
            throw MLXRuntimeDiagnosticsError.cacheLimitTooLarge(bytes)
        }
        Memory.cacheLimit = Int(bytes)
    }
}
