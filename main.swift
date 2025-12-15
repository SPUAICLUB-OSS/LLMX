import Charts
import Combine
import SwiftUI

@main
struct LLMXApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 1100, minHeight: 700)
        }
        .windowStyle(.hiddenTitleBar)
        .windowResizability(.contentSize)
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        NSWindow.allowsAutomaticWindowTabbing = false
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

struct ContentView: View {
    @StateObject private var viewModel = TrainingViewModel()
    @State private var selectedTab: Tab = .train

    enum Tab: String, CaseIterable {
        case train = "Train"
        case monitor = "Monitor"
        case export = "Export"
    }

    var body: some View {
        HStack(spacing: 0) {
            SidebarView(selectedTab: $selectedTab, viewModel: viewModel)
            Divider()
            MainContentView(selectedTab: selectedTab, viewModel: viewModel)
        }
        .background(Color(NSColor.windowBackgroundColor))
    }
}

struct SidebarView: View {
    @Binding var selectedTab: ContentView.Tab
    @ObservedObject var viewModel: TrainingViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            VStack(alignment: .leading, spacing: 4) {
                Text("LLMX")
                    .font(.system(size: 24, weight: .bold, design: .rounded))
                Text("SPU AI CLUB")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundColor(.secondary)
            }
            .padding(.horizontal, 20)
            .padding(.top, 40)
            .padding(.bottom, 30)

            ForEach(ContentView.Tab.allCases, id: \.self) { tab in
                SidebarButton(
                    title: tab.rawValue, icon: iconFor(tab), isSelected: selectedTab == tab
                ) {
                    withAnimation(.easeInOut(duration: 0.2)) { selectedTab = tab }
                }
            }

            Spacer()

            VStack(alignment: .leading, spacing: 8) {
                StatusRow(
                    color: viewModel.isTraining ? .orange : .green,
                    text: viewModel.isTraining ? "Training..." : "Ready")
                StatusRow(color: .green, text: "Apple Silicon")
                StatusRow(color: .blue, text: "MLX + CoreML")
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 20)
        }
        .frame(width: 200)
        .background(Color(NSColor.controlBackgroundColor).opacity(0.5))
    }

    private func iconFor(_ tab: ContentView.Tab) -> String {
        switch tab {
        case .train: return "cpu"
        case .monitor: return "chart.line.uptrend.xyaxis"
        case .export: return "square.and.arrow.up"
        }
    }
}

struct StatusRow: View {
    let color: Color
    let text: String
    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(color).frame(width: 8, height: 8)
            Text(text).font(.system(size: 11)).foregroundColor(.secondary)
        }
    }
}

struct SidebarButton: View {
    let title: String
    let icon: String
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: icon).font(.system(size: 14, weight: .medium)).frame(width: 20)
                Text(title).font(.system(size: 13, weight: isSelected ? .semibold : .regular))
                Spacer()
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 8).fill(
                    isSelected ? Color.accentColor.opacity(0.15) : Color.clear)
            )
            .foregroundColor(isSelected ? .accentColor : .primary)
        }
        .buttonStyle(.plain)
        .padding(.horizontal, 12)
        .padding(.vertical, 2)
    }
}

struct MainContentView: View {
    let selectedTab: ContentView.Tab
    @ObservedObject var viewModel: TrainingViewModel

    var body: some View {
        switch selectedTab {
        case .train: TrainView(viewModel: viewModel)
        case .monitor: MonitorView(viewModel: viewModel)
        case .export: ExportView(viewModel: viewModel)
        }
    }
}

enum ModelType: String, CaseIterable {
    case llm = "LLM"
    case imageClassification = "Image Classification"
    case objectDetection = "Object Detection"
    var icon: String {
        switch self {
        case .llm: return "text.alignleft"
        case .imageClassification: return "photo.stack"
        case .objectDetection: return "viewfinder"
        }
    }
}

enum BaseModel: String, CaseIterable {
    case custom = "Custom"
    case resnet50 = "ResNet-50"
    case resnet101 = "ResNet-101"
    case vgg16 = "VGG-16"
    case efficientnet = "EfficientNet"
    case mobilenet = "MobileNet"
    case vit = "ViT"
    case yolov8n = "YOLOv8n"
    case yolov8s = "YOLOv8s"
    case yolov8m = "YOLOv8m"
}

struct TrainView: View {
    @ObservedObject var viewModel: TrainingViewModel
    @State private var modelType: ModelType = .imageClassification
    @State private var baseModel: BaseModel = .resnet50
    @State private var modelPath = ""
    @State private var dataPath = ""
    @State private var epochs: Double = 10
    @State private var batchSize: Double = 32
    @State private var learningRate = "1e-4"
    @State private var imageSize: Double = 224
    @State private var numClasses: Double = 10
    @State private var augmentation = true
    @State private var pretrained = true

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 32) {
                HeaderView(
                    title: "Train Model", subtitle: "Configure and train using Apple MLX & CoreML")

                ConfigSection(title: "Model Type") {
                    HStack(spacing: 12) {
                        ForEach(ModelType.allCases, id: \.self) { type in
                            ModelTypeButton(
                                title: type.rawValue, icon: type.icon, isSelected: modelType == type
                            ) { modelType = type }
                        }
                    }
                }

                if modelType != .llm {
                    ConfigSection(title: "Base Model") {
                        LazyVGrid(
                            columns: Array(repeating: GridItem(.flexible()), count: 4), spacing: 10
                        ) {
                            ForEach(filteredModels, id: \.self) { model in
                                BaseModelButton(
                                    title: model.rawValue, isSelected: baseModel == model
                                ) { baseModel = model }
                            }
                        }
                        Toggle("Use Pretrained Weights", isOn: $pretrained).font(.system(size: 13))
                            .padding(.top, 8)
                    }
                }

                ConfigSection(title: "Data Configuration") {
                    if modelType == .llm {
                        PathSelector(
                            label: "Model Path", path: $modelPath,
                            placeholder: "Select model or HuggingFace ID")
                    }
                    PathSelector(
                        label: "Dataset Path", path: $dataPath,
                        placeholder: modelType == .llm
                            ? "Select training data folder"
                            : "Select images folder (class subfolders)", isDirectory: true)
                    if modelType != .llm {
                        SliderInput(
                            label: "Number of Classes", value: $numClasses, range: 2...1000, step: 1
                        )
                        SliderInput(
                            label: "Image Size", value: $imageSize, range: 32...640, step: 32)
                        Toggle("Data Augmentation", isOn: $augmentation).font(.system(size: 13))
                    }
                }

                ConfigSection(title: "Training Parameters") {
                    SliderInput(label: "Epochs", value: $epochs, range: 1...200, step: 1)
                    SliderInput(label: "Batch Size", value: $batchSize, range: 1...256, step: 1)
                    TextInput(label: "Learning Rate", text: $learningRate, placeholder: "1e-4")
                }

                HStack {
                    Spacer()
                    ActionButton(
                        title: viewModel.isTraining ? "Training..." : "Start Training",
                        isLoading: viewModel.isTraining, color: .accentColor
                    ) {
                        viewModel.startTraining(
                            modelType: modelType.rawValue, baseModel: baseModel.rawValue,
                            modelPath: modelPath, dataPath: dataPath, epochs: Int(epochs),
                            batchSize: Int(batchSize), learningRate: learningRate,
                            imageSize: Int(imageSize), numClasses: Int(numClasses),
                            augmentation: augmentation, pretrained: pretrained)
                    }.disabled(viewModel.isTraining)
                    if viewModel.isTraining {
                        ActionButton(title: "Stop", isLoading: false, color: .red) {
                            viewModel.stopTraining()
                        }
                    }
                }
            }
            .padding(40)
        }
    }

    private var filteredModels: [BaseModel] {
        switch modelType {
        case .llm: return [.custom]
        case .imageClassification:
            return [.custom, .resnet50, .resnet101, .vgg16, .efficientnet, .mobilenet, .vit]
        case .objectDetection: return [.yolov8n, .yolov8s, .yolov8m]
        }
    }
}

struct MonitorView: View {
    @ObservedObject var viewModel: TrainingViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                HeaderView(title: "Training Monitor", subtitle: "Real-time metrics and performance")
                    .padding(.horizontal, 40).padding(.top, 40)

                HStack(spacing: 16) {
                    MetricCard(
                        title: "Loss", value: String(format: "%.4f", viewModel.currentLoss),
                        icon: "chart.line.downtrend.xyaxis", color: .orange)
                    MetricCard(
                        title: "Accuracy",
                        value: String(format: "%.1f%%", viewModel.currentAccuracy * 100),
                        icon: "checkmark.circle", color: .green)
                    MetricCard(
                        title: "Epoch", value: "\(viewModel.currentEpoch)/\(viewModel.totalEpochs)",
                        icon: "arrow.trianglehead.2.clockwise.rotate.90", color: .blue)
                    MetricCard(
                        title: "Speed",
                        value: String(format: "%.1f it/s", viewModel.iterationsPerSecond),
                        icon: "bolt.fill", color: .purple)
                    MetricCard(
                        title: "Memory", value: String(format: "%.1f GB", viewModel.memoryUsage),
                        icon: "memorychip", color: .pink)
                }.padding(.horizontal, 40)

                HStack(spacing: 16) {
                    ChartCard(title: "Loss", data: viewModel.lossHistory, color: .orange)
                    ChartCard(
                        title: "Accuracy", data: viewModel.accuracyHistory, color: .green,
                        isPercentage: true)
                }.padding(.horizontal, 40)

                HStack(spacing: 16) {
                    ChartCard(title: "Learning Rate", data: viewModel.lrHistory, color: .blue)
                    ChartCard(
                        title: "GPU Memory (GB)", data: viewModel.memoryHistory, color: .purple)
                }.padding(.horizontal, 40)

                LogView(logs: $viewModel.logs).padding(.horizontal, 40).padding(.bottom, 40)
            }
        }
    }
}

struct ExportView: View {
    @ObservedObject var viewModel: TrainingViewModel
    @State private var exportPath = ""
    @State private var selectedFormats: Set<String> = ["CoreML (.mlmodel)"]

    let formats = [
        "CoreML (.mlmodel)", "MLX (.safetensors)", "PyTorch (.pt)", "ONNX (.onnx)", "Keras (.h5)",
        "GGUF (.gguf)",
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 32) {
                HeaderView(
                    title: "Export Model", subtitle: "Save your trained model in multiple formats")

                ConfigSection(title: "Export Destination") {
                    PathSelector(
                        label: "Export Path", path: $exportPath,
                        placeholder: "Select export folder", isDirectory: true)
                }

                ConfigSection(title: "Export Formats") {
                    Text("Select one or more formats").font(.system(size: 12)).foregroundColor(
                        .secondary)
                    LazyVGrid(
                        columns: Array(repeating: GridItem(.flexible()), count: 3), spacing: 12
                    ) {
                        ForEach(formats, id: \.self) { format in
                            FormatToggle(
                                format: format, isSelected: selectedFormats.contains(format)
                            ) {
                                if selectedFormats.contains(format) {
                                    selectedFormats.remove(format)
                                } else {
                                    selectedFormats.insert(format)
                                }
                            }
                        }
                    }
                }

                ConfigSection(title: "Model Info") {
                    HStack(spacing: 40) {
                        VStack(alignment: .leading, spacing: 8) {
                            InfoRow(label: "Model Type", value: viewModel.modelType)
                            InfoRow(label: "Base Model", value: viewModel.baseModel)
                            InfoRow(label: "Parameters", value: viewModel.parameterCount)
                        }
                        VStack(alignment: .leading, spacing: 8) {
                            InfoRow(
                                label: "Final Loss",
                                value: String(format: "%.4f", viewModel.currentLoss))
                            InfoRow(
                                label: "Final Accuracy",
                                value: String(format: "%.1f%%", viewModel.currentAccuracy * 100))
                            InfoRow(label: "Training Time", value: viewModel.trainingTime)
                        }
                    }
                }

                HStack {
                    Spacer()
                    ActionButton(
                        title: viewModel.isExporting ? "Exporting..." : "Export Model",
                        isLoading: viewModel.isExporting, color: .accentColor
                    ) {
                        viewModel.exportModel(to: exportPath, formats: Array(selectedFormats))
                    }.disabled(!canExport)
                }
            }
            .padding(40)
        }
    }

    private var canExport: Bool {
        !exportPath.isEmpty && !selectedFormats.isEmpty && viewModel.hasTrainedModel
            && !viewModel.isExporting
    }
}

struct HeaderView: View {
    let title: String
    let subtitle: String
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title).font(.system(size: 28, weight: .bold, design: .rounded))
            Text(subtitle).font(.system(size: 14)).foregroundColor(.secondary)
        }
    }
}

struct ModelTypeButton: View {
    let title: String
    let icon: String
    let isSelected: Bool
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            VStack(spacing: 8) {
                Image(systemName: icon).font(.system(size: 24))
                Text(title).font(.system(size: 11, weight: .medium))
            }
            .frame(maxWidth: .infinity).padding(.vertical, 16)
            .background(
                RoundedRectangle(cornerRadius: 10).fill(
                    isSelected ? Color.accentColor : Color(NSColor.controlBackgroundColor))
            )
            .foregroundColor(isSelected ? .white : .primary)
        }.buttonStyle(.plain)
    }
}

struct BaseModelButton: View {
    let title: String
    let isSelected: Bool
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            Text(title).font(.system(size: 12, weight: isSelected ? .semibold : .regular))
                .frame(maxWidth: .infinity).padding(.vertical, 10)
                .background(
                    RoundedRectangle(cornerRadius: 8).fill(
                        isSelected ? Color.accentColor : Color(NSColor.controlBackgroundColor))
                )
                .foregroundColor(isSelected ? .white : .primary)
        }.buttonStyle(.plain)
    }
}

struct ActionButton: View {
    let title: String
    let isLoading: Bool
    let color: Color
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                if isLoading {
                    ProgressView().scaleEffect(0.7).progressViewStyle(
                        CircularProgressViewStyle(tint: .white))
                }
                Text(title).font(.system(size: 14, weight: .semibold))
            }
            .frame(width: 160, height: 44)
            .background(RoundedRectangle(cornerRadius: 10).fill(isLoading ? Color.gray : color))
            .foregroundColor(.white)
        }.buttonStyle(.plain)
    }
}

struct FormatToggle: View {
    let format: String
    let isSelected: Bool
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle").foregroundColor(
                    isSelected ? .accentColor : .secondary)
                Text(format).font(.system(size: 12))
                Spacer()
            }
            .padding(.horizontal, 12).padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 8).fill(
                    isSelected
                        ? Color.accentColor.opacity(0.1) : Color(NSColor.controlBackgroundColor)))
        }.buttonStyle(.plain)
    }
}

struct InfoRow: View {
    let label: String
    let value: String
    var body: some View {
        HStack {
            Text(label).font(.system(size: 12)).foregroundColor(.secondary)
            Spacer()
            Text(value).font(.system(size: 12, weight: .medium, design: .monospaced))
        }
    }
}

struct ConfigSection<Content: View>: View {
    let title: String
    @ViewBuilder let content: Content
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(title).font(.system(size: 16, weight: .semibold))
            VStack(spacing: 16) { content }
                .padding(20)
                .background(
                    RoundedRectangle(cornerRadius: 12).fill(
                        Color(NSColor.controlBackgroundColor).opacity(0.5)))
        }
    }
}

struct PathSelector: View {
    let label: String
    @Binding var path: String
    let placeholder: String
    var isDirectory = false
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(label).font(.system(size: 13, weight: .medium)).foregroundColor(.secondary)
            HStack(spacing: 12) {
                TextField(placeholder, text: $path).textFieldStyle(.plain).font(.system(size: 13))
                    .padding(.horizontal, 12).padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: 8).fill(Color(NSColor.textBackgroundColor)))
                Button(action: selectPath) {
                    Image(systemName: "folder").font(.system(size: 14)).frame(width: 36, height: 36)
                        .background(
                            RoundedRectangle(cornerRadius: 8).fill(
                                Color(NSColor.controlBackgroundColor)))
                }.buttonStyle(.plain)
            }
        }
    }
    private func selectPath() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = isDirectory
        panel.canChooseFiles = !isDirectory
        if panel.runModal() == .OK { path = panel.url?.path ?? "" }
    }
}

struct SliderInput: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let step: Double
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(label).font(.system(size: 13, weight: .medium)).foregroundColor(.secondary)
                Spacer()
                Text("\(Int(value))").font(
                    .system(size: 13, weight: .semibold, design: .monospaced))
            }
            Slider(value: $value, in: range, step: step).tint(.accentColor)
        }
    }
}

struct TextInput: View {
    let label: String
    @Binding var text: String
    let placeholder: String
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(label).font(.system(size: 13, weight: .medium)).foregroundColor(.secondary)
            TextField(placeholder, text: $text).textFieldStyle(.plain).font(
                .system(size: 13, design: .monospaced)
            )
            .padding(.horizontal, 12).padding(.vertical, 10)
            .background(RoundedRectangle(cornerRadius: 8).fill(Color(NSColor.textBackgroundColor)))
        }
    }
}

struct MetricCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Image(systemName: icon).font(.system(size: 16)).foregroundColor(color)
            VStack(alignment: .leading, spacing: 4) {
                Text(value).font(.system(size: 20, weight: .bold, design: .rounded))
                Text(title).font(.system(size: 11)).foregroundColor(.secondary)
            }
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12).fill(
                Color(NSColor.controlBackgroundColor).opacity(0.5)))
    }
}

struct ChartCard: View {
    let title: String
    let data: [Double]
    let color: Color
    var isPercentage = false
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(title).font(.system(size: 14, weight: .semibold))
                Spacer()
                if let last = data.last {
                    Text(
                        isPercentage
                            ? String(format: "%.1f%%", last * 100) : String(format: "%.4f", last)
                    )
                    .font(.system(size: 12, design: .monospaced)).foregroundColor(.secondary)
                }
            }
            if data.isEmpty {
                Rectangle().fill(Color.secondary.opacity(0.2)).frame(height: 120)
                    .overlay(Text("No data").font(.system(size: 12)).foregroundColor(.secondary))
            } else {
                Chart(Array(data.enumerated()), id: \.offset) { item in
                    LineMark(
                        x: .value("Step", item.offset),
                        y: .value("Value", isPercentage ? item.element * 100 : item.element)
                    )
                    .foregroundStyle(color.gradient).interpolationMethod(.catmullRom)
                    AreaMark(
                        x: .value("Step", item.offset),
                        y: .value("Value", isPercentage ? item.element * 100 : item.element)
                    )
                    .foregroundStyle(color.opacity(0.1).gradient).interpolationMethod(.catmullRom)
                }
                .frame(height: 120).chartXAxis(.hidden)
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 12).fill(
                Color(NSColor.controlBackgroundColor).opacity(0.5)))
    }
}

struct LogView: View {
    @Binding var logs: [String]
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Training Log").font(.system(size: 16, weight: .semibold))
                Spacer()
                Button("Clear") { logs.removeAll() }.font(.system(size: 12)).foregroundColor(
                    .secondary
                ).buttonStyle(.plain)
            }
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 4) {
                        ForEach(logs.indices, id: \.self) { i in
                            Text(logs[i]).font(.system(size: 11, design: .monospaced))
                                .foregroundColor(logColor(logs[i])).id(i)
                        }
                    }.padding(16)
                }
                .frame(height: 200)
                .background(
                    RoundedRectangle(cornerRadius: 12).fill(Color(NSColor.textBackgroundColor))
                )
                .onChange(of: logs.count) { _ in
                    withAnimation { proxy.scrollTo(logs.count - 1, anchor: .bottom) }
                }
            }
        }
    }
    private func logColor(_ text: String) -> Color {
        if text.contains("[ERROR]") { return .red }
        if text.contains("[WARN]") { return .orange }
        if text.contains("[DONE]") { return .green }
        if text.contains("[START]") { return .blue }
        return .secondary
    }
}

class TrainingViewModel: ObservableObject {
    @Published var isTraining = false
    @Published var isExporting = false
    @Published var currentLoss = 0.0
    @Published var currentAccuracy = 0.0
    @Published var currentEpoch = 0
    @Published var totalEpochs = 0
    @Published var iterationsPerSecond = 0.0
    @Published var memoryUsage = 0.0
    @Published var logs: [String] = []
    @Published var hasTrainedModel = false
    @Published var lossHistory: [Double] = []
    @Published var accuracyHistory: [Double] = []
    @Published var lrHistory: [Double] = []
    @Published var memoryHistory: [Double] = []
    @Published var modelType = "-"
    @Published var baseModel = "-"
    @Published var parameterCount = "-"
    @Published var trainingTime = "-"

    private var linker = PythonLinker()
    private var startTime: Date?

    init() { setupCallbacks() }

    private func setupCallbacks() {
        linker.onLogReceived = { [weak self] log in
            DispatchQueue.main.async { self?.logs.append(log) }
        }
        linker.onMetricsUpdated = { [weak self] m in
            DispatchQueue.main.async {
                self?.currentLoss = m.loss
                self?.currentAccuracy = m.accuracy
                self?.currentEpoch = m.epoch
                self?.iterationsPerSecond = m.iterPerSec
                self?.memoryUsage = m.memoryGB
                self?.lossHistory.append(m.loss)
                self?.accuracyHistory.append(m.accuracy)
                self?.lrHistory.append(m.learningRate)
                self?.memoryHistory.append(m.memoryGB)
                if (self?.lossHistory.count ?? 0) > 500 {
                    self?.lossHistory.removeFirst()
                    self?.accuracyHistory.removeFirst()
                    self?.lrHistory.removeFirst()
                    self?.memoryHistory.removeFirst()
                }
            }
        }
        linker.onTrainingCompleted = { [weak self] success in
            DispatchQueue.main.async {
                self?.isTraining = false
                self?.hasTrainedModel = success
                if let start = self?.startTime {
                    self?.trainingTime =
                        self?.formatDuration(Date().timeIntervalSince(start)) ?? "-"
                }
                self?.logs.append(success ? "[DONE] Training completed" : "[ERROR] Training failed")
            }
        }
        linker.onModelInfo = { [weak self] info in
            DispatchQueue.main.async { self?.parameterCount = info.parameterCount }
        }
    }

    private func formatDuration(_ s: TimeInterval) -> String {
        let h = Int(s) / 3600
        let m = (Int(s) % 3600) / 60
        let sec = Int(s) % 60
        return h > 0 ? "\(h)h \(m)m" : m > 0 ? "\(m)m \(sec)s" : "\(sec)s"
    }

    func startTraining(
        modelType: String, baseModel: String, modelPath: String, dataPath: String, epochs: Int,
        batchSize: Int, learningRate: String, imageSize: Int, numClasses: Int, augmentation: Bool,
        pretrained: Bool
    ) {
        isTraining = true
        totalEpochs = epochs
        self.modelType = modelType
        self.baseModel = baseModel
        startTime = Date()
        lossHistory.removeAll()
        accuracyHistory.removeAll()
        lrHistory.removeAll()
        memoryHistory.removeAll()
        logs.removeAll()
        logs.append("[START] Initializing \(modelType) training...")
        linker.train(
            modelType: modelType, baseModel: baseModel, modelPath: modelPath, dataPath: dataPath,
            epochs: epochs, batchSize: batchSize, learningRate: learningRate, imageSize: imageSize,
            numClasses: numClasses, augmentation: augmentation, pretrained: pretrained)
    }

    func stopTraining() {
        linker.stop()
        isTraining = false
        logs.append("[STOP] Training stopped")
    }

    func exportModel(to path: String, formats: [String]) {
        isExporting = true
        logs.append("[EXPORT] Exporting to \(formats.joined(separator: ", "))...")
        linker.export(to: path, formats: formats) { [weak self] success in
            DispatchQueue.main.async {
                self?.isExporting = false
                self?.logs.append(success ? "[DONE] Export completed" : "[ERROR] Export failed")
            }
        }
    }
}
