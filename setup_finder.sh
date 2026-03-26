#!/usr/bin/env bash
# setup_finder.sh — macOS Finder クイックアクション自動生成
#
# Finder の右クリックメニューに以下を追加:
#   - 「APLZ で圧縮」: 選択したファイル/フォルダを .aplz / .tar.aplz に圧縮
#   - 「APLZ で解凍」: 選択した .aplz ファイルを解凍
#
# 生成先: ~/Library/Services/*.workflow
# 削除:   ./setup_finder.sh uninstall

set -euo pipefail

SERVICES_DIR="$HOME/Library/Services"

# aplz コマンドのパスを検出
find_aplz() {
    local candidates=(
        "/usr/local/bin/aplz"
        "$HOME/.local/bin/aplz"
        "$(cd "$(dirname "$0")" && pwd)/aplz"
    )
    for c in "${candidates[@]}"; do
        [[ -x "$c" ]] && echo "$c" && return
    done
    echo ""
}

APLZ_PATH="$(find_aplz)"
if [[ -z "$APLZ_PATH" ]]; then
    echo "Error: aplz command not found." >&2
    echo "Run install.sh first, then re-run this script." >&2
    exit 1
fi

echo "==> Using aplz at: ${APLZ_PATH}"

# ─── ワークフロー生成 ────────────────────────────────────────────────────────
create_workflow() {
    local name="$1"
    local script="$2"
    local wf_dir="${SERVICES_DIR}/${name}.workflow/Contents"

    mkdir -p "$wf_dir"

    # Info.plist
    cat > "${wf_dir}/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>NSServices</key>
    <array>
        <dict>
            <key>NSMenuItem</key>
            <dict>
                <key>default</key>
                <string>APLZ_SERVICE_NAME</string>
            </dict>
            <key>NSMessage</key>
            <string>runWorkflowAsService</string>
        </dict>
    </array>
</dict>
</plist>
PLIST
    # サービス名を置換
    sed -i '' "s/APLZ_SERVICE_NAME/${name}/" "${wf_dir}/Info.plist"

    # document.wflow (Automator workflow XML)
    cat > "${wf_dir}/document.wflow" <<WFLOW
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>AMApplicationBuild</key>
    <string>523</string>
    <key>AMApplicationVersion</key>
    <string>2.10</string>
    <key>AMDocumentVersion</key>
    <string>2</string>
    <key>actions</key>
    <array>
        <dict>
            <key>action</key>
            <dict>
                <key>AMAccepts</key>
                <dict>
                    <key>Container</key>
                    <string>List</string>
                    <key>Optional</key>
                    <false/>
                    <key>Types</key>
                    <array>
                        <string>com.apple.cocoa.path</string>
                    </array>
                </dict>
                <key>AMActionVersion</key>
                <string>1.0.2</string>
                <key>AMApplication</key>
                <array>
                    <string>Automator</string>
                </array>
                <key>AMBundleIdentifier</key>
                <string>com.apple.RunShellScript</string>
                <key>AMCategory</key>
                <array>
                    <string>AMCategoryUtilities</string>
                </array>
                <key>AMIconName</key>
                <string>Automator</string>
                <key>AMKeywords</key>
                <array>
                    <string>Shell</string>
                    <string>Script</string>
                </array>
                <key>AMName</key>
                <string>Run Shell Script</string>
                <key>AMProvides</key>
                <dict>
                    <key>Container</key>
                    <string>List</string>
                    <key>Types</key>
                    <array>
                        <string>com.apple.cocoa.path</string>
                    </array>
                </dict>
                <key>ActionBundlePath</key>
                <string>/System/Library/Automator/Run Shell Script.action</string>
                <key>ActionName</key>
                <string>Run Shell Script</string>
                <key>ActionParameters</key>
                <dict>
                    <key>COMMAND_STRING</key>
                    <string>${script}</string>
                    <key>CheckedForUserDefaultShell</key>
                    <true/>
                    <key>inputMethod</key>
                    <integer>1</integer>
                    <key>shell</key>
                    <string>/bin/bash</string>
                    <key>source</key>
                    <string></string>
                </dict>
                <key>BundleIdentifier</key>
                <string>com.apple.RunShellScript</string>
                <key>CFBundleVersion</key>
                <string>1.0.2</string>
                <key>CanShowSelectedItemsWhenRun</key>
                <true/>
                <key>CanShowWhenRun</key>
                <true/>
                <key>Category</key>
                <array>
                    <string>AMCategoryUtilities</string>
                </array>
                <key>Class Name</key>
                <string>RunShellScriptAction</string>
                <key>InputUUID</key>
                <string>$(uuidgen)</string>
                <key>Keywords</key>
                <array>
                    <string>Shell</string>
                    <string>Script</string>
                </array>
                <key>OutputUUID</key>
                <string>$(uuidgen)</string>
                <key>UUID</key>
                <string>$(uuidgen)</string>
                <key>UnlocalizedApplications</key>
                <array>
                    <string>Automator</string>
                </array>
            </dict>
        </dict>
    </array>
    <key>connectors</key>
    <dict/>
    <key>workflowMetaData</key>
    <dict>
        <key>applicationBundleIDsByPath</key>
        <dict/>
        <key>applicationPaths</key>
        <array/>
        <key>inputTypeIdentifier</key>
        <string>com.apple.Automator.fileSystemObject</string>
        <key>outputTypeIdentifier</key>
        <string>com.apple.Automator.nothing</string>
        <key>presentationMode</key>
        <integer>15</integer>
        <key>processesInput</key>
        <integer>0</integer>
        <key>serviceApplicationPath</key>
        <string>/System/Library/CoreServices/Finder.app</string>
        <key>serviceInputTypeIdentifier</key>
        <string>com.apple.Automator.fileSystemObject</string>
        <key>serviceProcessesInput</key>
        <integer>0</integer>
        <key>systemImageName</key>
        <string>NSActionTemplate</string>
        <key>useAutomaticInputType</key>
        <integer>0</integer>
        <key>workflowTypeIdentifier</key>
        <string>com.apple.Automator.servicesMenu</string>
    </dict>
</dict>
</plist>
WFLOW
}

# ─── メイン処理 ──────────────────────────────────────────────────────────────

if [[ "${1:-}" == "uninstall" ]]; then
    echo "==> Removing Finder quick actions..."
    rm -rf "${SERVICES_DIR}/APLZ で圧縮.workflow"
    rm -rf "${SERVICES_DIR}/APLZ で解凍.workflow"
    echo "Done. Quick actions removed."
    exit 0
fi

mkdir -p "$SERVICES_DIR"

# 圧縮アクション
COMPRESS_SCRIPT="for f in \"\$@\"; do
    \"${APLZ_PATH}\" compress \"\$f\"
    osascript -e 'display notification \"圧縮完了: '\"\$(basename \"\$f\")\"'\" with title \"APLZ\"'
done"

echo "==> Creating: APLZ で圧縮.workflow"
create_workflow "APLZ で圧縮" "$COMPRESS_SCRIPT"

# 解凍アクション
EXTRACT_SCRIPT="for f in \"\$@\"; do
    dir=\$(dirname \"\$f\")
    \"${APLZ_PATH}\" extract \"\$f\" \"\$dir\"
    osascript -e 'display notification \"解凍完了: '\"\$(basename \"\$f\")\"'\" with title \"APLZ\"'
done"

echo "==> Creating: APLZ で解凍.workflow"
create_workflow "APLZ で解凍" "$EXTRACT_SCRIPT"

echo ""
echo "============================================================"
echo " Finder クイックアクションをインストールしました"
echo ""
echo " 使い方:"
echo "   1. Finder でファイル/フォルダを右クリック"
echo "   2.「クイックアクション」→「APLZ で圧縮」を選択"
echo "   3. .aplz または .tar.aplz ファイルを右クリック"
echo "   4.「クイックアクション」→「APLZ で解凍」を選択"
echo ""
echo " アンインストール:"
echo "   ./setup_finder.sh uninstall"
echo "============================================================"
