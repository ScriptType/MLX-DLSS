import AppKit
import SwiftUI

@main
enum AppLauncher {
  static func main() {
    guard #available(macOS 26.0, *) else {
      fputs("MLX DLSS requires macOS 26 or newer.\n", stderr)
      return
    }
    NSApplication.shared.setActivationPolicy(.regular)
    MLXDLSSApplication.main()
  }
}

@available(macOS 26.0, *)
struct MLXDLSSApplication: App {
  @State private var model = AppModel()

  var body: some Scene {
    Window("MLX DLSS", id: "main") {
      MainView(model: model)
        .frame(minWidth: 980, minHeight: 680)
        .onOpenURL { model.addFiles([$0]) }
        .onAppear { NSApplication.shared.activate() }
    }
    .defaultSize(width: 1240, height: 820)
    .commands {
      CommandGroup(replacing: .newItem) {
        Button("Import Images or Video…") { model.chooseFiles() }.keyboardShortcut("o")
      }
    }
  }
}
