import Foundation

struct TrainingMetrics {
    var loss: Double = 0.0
    var accuracy: Double = 0.0
    var epoch: Int = 0
    var step: Int = 0
    var iterPerSec: Double = 0.0
    var memoryGB: Double = 0.0
    var learningRate: Double = 0.0
}

struct ModelInfo {
    var parameterCount: String = "-"
}

class PythonLinker {
    private var process: Process?
    private var inputPipe: Pipe?
    private var outputPipe: Pipe?
    private var isRunning = false
    private let queue = DispatchQueue(label: "com.llmx.linker", qos: .userInitiated)

    var onLogReceived: ((String) -> Void)?
    var onMetricsUpdated: ((TrainingMetrics) -> Void)?
    var onTrainingCompleted: ((Bool) -> Void)?
    var onModelInfo: ((ModelInfo) -> Void)?

    private var pythonPath: String {
        let paths = [
            "/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3",
            Bundle.main.resourcePath.map { "\($0)/python3" } ?? "",
        ]
        for path in paths { if FileManager.default.fileExists(atPath: path) { return path } }
        return "python3"
    }

    private var scriptPath: String {
        if let resourcePath = Bundle.main.resourcePath {
            let bundled = "\(resourcePath)/train.py"
            if FileManager.default.fileExists(atPath: bundled) { return bundled }
        }
        return "\(FileManager.default.currentDirectoryPath)/train.py"
    }

    func train(
        modelType: String, baseModel: String, modelPath: String, dataPath: String, epochs: Int,
        batchSize: Int, learningRate: String, imageSize: Int, numClasses: Int, augmentation: Bool,
        pretrained: Bool
    ) {
        queue.async { [weak self] in
            self?.startProcess()
            let command: [String: Any] = [
                "action": "train",
                "model_type": modelType,
                "base_model": baseModel,
                "model_path": modelPath,
                "data_path": dataPath,
                "epochs": epochs,
                "batch_size": batchSize,
                "learning_rate": learningRate,
                "image_size": imageSize,
                "num_classes": numClasses,
                "augmentation": augmentation,
                "pretrained": pretrained,
            ]
            self?.sendCommand(command)
        }
    }

    func stop() {
        queue.async { [weak self] in
            self?.sendCommand(["action": "stop"])
        }
    }

    func export(to path: String, formats: [String], completion: @escaping (Bool) -> Void) {
        queue.async { [weak self] in
            let command: [String: Any] = [
                "action": "export", "output_path": path, "formats": formats,
            ]
            self?.sendCommand(command)
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) { completion(true) }
        }
    }

    private func startProcess() {
        guard !isRunning else { return }

        process = Process()
        inputPipe = Pipe()
        outputPipe = Pipe()

        process?.executableURL = URL(fileURLWithPath: pythonPath)
        process?.arguments = ["-u", scriptPath, "--server"]
        process?.standardInput = inputPipe
        process?.standardOutput = outputPipe
        process?.standardError = FileHandle.nullDevice

        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env["MLX_USE_GPU"] = "1"
        process?.environment = env

        outputPipe?.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            if !data.isEmpty { self?.handleOutput(data) }
        }

        process?.terminationHandler = { [weak self] _ in
            DispatchQueue.main.async { self?.isRunning = false }
        }

        do {
            try process?.run()
            isRunning = true
        } catch {
            DispatchQueue.main.async { [weak self] in
                self?.onLogReceived?("[ERROR] Failed to start: \(error.localizedDescription)")
            }
        }
    }

    private func sendCommand(_ command: [String: Any]) {
        guard let inputPipe = inputPipe else { return }
        do {
            var data = try JSONSerialization.data(withJSONObject: command)
            data.append(contentsOf: "\n".utf8)
            inputPipe.fileHandleForWriting.write(data)
        } catch {
            DispatchQueue.main.async { [weak self] in
                self?.onLogReceived?("[ERROR] Command failed: \(error.localizedDescription)")
            }
        }
    }

    private func handleOutput(_ data: Data) {
        guard let string = String(data: data, encoding: .utf8) else { return }
        for line in string.components(separatedBy: "\n").filter({ !$0.isEmpty }) {
            processMessage(line)
        }
    }

    private func processMessage(_ line: String) {
        guard let data = line.data(using: .utf8),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let type = json["type"] as? String
        else { return }

        DispatchQueue.main.async { [weak self] in
            switch type {
            case "ready":
                self?.onLogReceived?("[SYSTEM] Backend ready")
            case "log":
                if let msg = json["message"] as? String { self?.onLogReceived?(msg) }
            case "metrics":
                var m = TrainingMetrics()
                m.loss = json["loss"] as? Double ?? 0.0
                m.accuracy = json["accuracy"] as? Double ?? 0.0
                m.epoch = json["epoch"] as? Int ?? 0
                m.step = json["step"] as? Int ?? 0
                m.iterPerSec = json["iter_per_sec"] as? Double ?? 0.0
                m.memoryGB = json["memory_gb"] as? Double ?? 0.0
                m.learningRate = json["learning_rate"] as? Double ?? 0.0
                self?.onMetricsUpdated?(m)
            case "completed":
                let success = json["success"] as? Bool ?? false
                self?.onTrainingCompleted?(success)
                self?.terminateProcess()
            case "model_info":
                var info = ModelInfo()
                info.parameterCount = json["parameter_count"] as? String ?? "-"
                self?.onModelInfo?(info)
            default:
                break
            }
        }
    }

    private func terminateProcess() {
        outputPipe?.fileHandleForReading.readabilityHandler = nil
        process?.terminate()
        process = nil
        inputPipe = nil
        outputPipe = nil
        isRunning = false
    }

    deinit { terminateProcess() }
}

class ModelManager {
    static let shared = ModelManager()
    private let fm = FileManager.default

    var modelsDirectory: URL {
        let dir = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("LLMX/Models")
        try? fm.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func listModels() -> [URL] {
        (try? fm.contentsOfDirectory(at: modelsDirectory, includingPropertiesForKeys: nil))?.filter
        {
            [".safetensors", ".gguf", ".mlmodel", ".pt", ".onnx", ".h5"].contains($0.pathExtension)
        } ?? []
    }

    func modelSize(at url: URL) -> String {
        guard let attrs = try? fm.attributesOfItem(atPath: url.path),
            let size = attrs[.size] as? Int64
        else { return "Unknown" }
        return ByteCountFormatter.string(fromByteCount: size, countStyle: .file)
    }
}
