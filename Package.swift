// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "CantooCapacitorOnnx",
    platforms: [.iOS(.v14)],
    products: [
        .library(
            name: "CantooCapacitorOnnx",
            targets: ["CapacitorOnnxPlugin"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ionic-team/capacitor-swift-pm", from: "7.0.0"),
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", from: "1.24.2"),
    ],
    targets: [
        .target(
            name: "CapacitorOnnxPlugin",
            dependencies: [
                .product(name: "Capacitor", package: "capacitor-swift-pm"),
                .product(name: "Cordova", package: "capacitor-swift-pm"),
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ],
            path: "ios/Plugin"
        )
    ]
)
