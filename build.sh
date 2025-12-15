#!/bin/bash

set -e

APP_NAME="LLMX"
BUNDLE_ID="com.spuaiclub.llmx"
VERSION="1.0.0"

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
APP_DIR="$BUILD_DIR/$APP_NAME.app"
CONTENTS_DIR="$APP_DIR/Contents"
MACOS_DIR="$CONTENTS_DIR/MacOS"
RESOURCES_DIR="$CONTENTS_DIR/Resources"

rm -rf "$BUILD_DIR"
mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

cat > "$CONTENTS_DIR/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundleDisplayName</key>
    <string>LLMX by SPU AI CLUB</string>
    <key>CFBundleVersion</key>
    <string>$VERSION</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>LSArchitecturePriority</key>
    <array>
        <string>arm64</string>
    </array>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.developer-tools</string>
</dict>
</plist>
EOF

echo "Compiling Swift sources..."
swiftc \
    -O \
    -whole-module-optimization \
    -parse-as-library \
    -target arm64-apple-macos13.0 \
    -sdk $(xcrun --sdk macosx --show-sdk-path) \
    -framework SwiftUI \
    -framework AppKit \
    -framework Foundation \
    -framework Combine \
    -o "$MACOS_DIR/$APP_NAME" \
    "$PROJECT_DIR/Sources/main.swift" \
    "$PROJECT_DIR/Sources/linker.swift"

cp "$PROJECT_DIR/Sources/train.py" "$RESOURCES_DIR/"

if [ -f "$PROJECT_DIR/Resources/AppIcon.icns" ]; then
    cp "$PROJECT_DIR/Resources/AppIcon.icns" "$RESOURCES_DIR/"
fi

chmod +x "$MACOS_DIR/$APP_NAME"
chmod +x "$RESOURCES_DIR/train.py"

echo "Creating DMG..."
DMG_DIR="$BUILD_DIR/dmg"
DMG_PATH="$BUILD_DIR/$APP_NAME-$VERSION.dmg"

mkdir -p "$DMG_DIR"
cp -R "$APP_DIR" "$DMG_DIR/"
ln -s /Applications "$DMG_DIR/Applications"

hdiutil create \
    -volname "$APP_NAME" \
    -srcfolder "$DMG_DIR" \
    -ov \
    -format UDZO \
    "$DMG_PATH"

rm -rf "$DMG_DIR"

echo ""
echo "Build completed successfully!"
echo "App: $APP_DIR"
echo "DMG: $DMG_PATH"
