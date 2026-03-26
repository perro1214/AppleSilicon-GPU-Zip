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
# 実績のある Automator Quick Action (servicesMenu) の document.wflow 形式で生成。
# 「Run Shell Script」アクションで、入力をコマンド引数として受け取る。
create_workflow() {
    local name="$1"
    local shell_script="$2"
    local wf_dir="${SERVICES_DIR}/${name}.workflow/Contents"

    mkdir -p "$wf_dir"

    local uuid1 uuid2 uuid3
    uuid1=$(uuidgen)
    uuid2=$(uuidgen)
    uuid3=$(uuidgen)

    # shell_script 内の & < > " を XML エスケープ
    local escaped_script
    escaped_script=$(echo "$shell_script" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g; s/"/\&quot;/g')

    cat > "${wf_dir}/document.wflow" <<WFLOW
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>AMApplicationBuild</key>
	<string>528</string>
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
					<true/>
					<key>Types</key>
					<array>
						<string>com.apple.cocoa.string</string>
					</array>
				</dict>
				<key>AMActionVersion</key>
				<string>2.0.3</string>
				<key>AMApplication</key>
				<array>
					<string>Automator</string>
				</array>
				<key>AMParameterProperties</key>
				<dict>
					<key>COMMAND_STRING</key>
					<dict/>
					<key>CheckedForUserDefaultShell</key>
					<dict/>
					<key>inputMethod</key>
					<dict/>
					<key>shell</key>
					<dict/>
					<key>source</key>
					<dict/>
				</dict>
				<key>AMProvides</key>
				<dict>
					<key>Container</key>
					<string>List</string>
					<key>Types</key>
					<array>
						<string>com.apple.cocoa.string</string>
					</array>
				</dict>
				<key>ActionBundlePath</key>
				<string>/System/Library/Automator/Run Shell Script.action</string>
				<key>ActionName</key>
				<string>シェルスクリプトを実行</string>
				<key>ActionParameters</key>
				<dict>
					<key>COMMAND_STRING</key>
					<string>${escaped_script}</string>
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
				<string>2.0.3</string>
				<key>CanShowSelectedItemsWhenRun</key>
				<false/>
				<key>CanShowWhenRun</key>
				<true/>
				<key>Category</key>
				<array>
					<string>AMCategoryUtilities</string>
				</array>
				<key>Class Name</key>
				<string>RunShellScriptAction</string>
				<key>InputUUID</key>
				<string>${uuid1}</string>
				<key>Keywords</key>
				<array>
					<string>Shell</string>
					<string>Script</string>
				</array>
				<key>OutputUUID</key>
				<string>${uuid2}</string>
				<key>UUID</key>
				<string>${uuid3}</string>
				<key>UnlocalizedApplications</key>
				<array>
					<string>Automator</string>
				</array>
				<key>arguments</key>
				<dict>
					<key>0</key>
					<dict>
						<key>default value</key>
						<string>for f in &quot;\$@&quot;; do echo &quot;\$f&quot;; done</string>
						<key>name</key>
						<string>COMMAND_STRING</string>
						<key>required</key>
						<string>0</string>
						<key>type</key>
						<string>0</string>
						<key>uuid</key>
						<string>0</string>
					</dict>
					<key>1</key>
					<dict>
						<key>default value</key>
						<string>/bin/bash</string>
						<key>name</key>
						<string>shell</string>
						<key>required</key>
						<string>0</string>
						<key>type</key>
						<string>0</string>
						<key>uuid</key>
						<string>1</string>
					</dict>
					<key>2</key>
					<dict>
						<key>default value</key>
						<integer>1</integer>
						<key>name</key>
						<string>inputMethod</string>
						<key>required</key>
						<string>0</string>
						<key>type</key>
						<string>0</string>
						<key>uuid</key>
						<string>2</string>
					</dict>
					<key>3</key>
					<dict>
						<key>default value</key>
						<true/>
						<key>name</key>
						<string>CheckedForUserDefaultShell</string>
						<key>required</key>
						<string>0</string>
						<key>type</key>
						<string>0</string>
						<key>uuid</key>
						<string>3</string>
					</dict>
				</dict>
				<key>conversionLabel</key>
				<integer>0</integer>
				<key>isViewVisible</key>
				<integer>1</integer>
				<key>location</key>
				<string>301.500000:253.000000</string>
				<key>nibPath</key>
				<string>/System/Library/Automator/Run Shell Script.action/Contents/Resources/Base.lproj/main.nib</string>
			</dict>
			<key>isViewVisible</key>
			<integer>1</integer>
		</dict>
	</array>
	<key>connectors</key>
	<dict/>
	<key>workflowMetaData</key>
	<dict>
		<key>applicationBundleID</key>
		<string>com.apple.finder</string>
		<key>applicationBundleIDsByPath</key>
		<dict>
			<key>/System/Library/CoreServices/Finder.app</key>
			<string>com.apple.finder</string>
		</dict>
		<key>applicationPath</key>
		<string>/System/Library/CoreServices/Finder.app</string>
		<key>applicationPaths</key>
		<array>
			<string>/System/Library/CoreServices/Finder.app</string>
		</array>
		<key>inputTypeIdentifier</key>
		<string>com.apple.Automator.fileSystemObject</string>
		<key>outputTypeIdentifier</key>
		<string>com.apple.Automator.nothing</string>
		<key>presentationMode</key>
		<integer>15</integer>
		<key>processesInput</key>
		<false/>
		<key>serviceApplicationBundleID</key>
		<string>com.apple.finder</string>
		<key>serviceApplicationPath</key>
		<string>/System/Library/CoreServices/Finder.app</string>
		<key>serviceInputTypeIdentifier</key>
		<string>com.apple.Automator.fileSystemObject</string>
		<key>serviceOutputTypeIdentifier</key>
		<string>com.apple.Automator.nothing</string>
		<key>serviceProcessesInput</key>
		<false/>
		<key>systemImageName</key>
		<string>NSActionTemplate</string>
		<key>useAutomaticInputType</key>
		<false/>
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

# 圧縮アクション: 選択されたファイル/ディレクトリを aplz で圧縮
COMPRESS_SCRIPT='for f in "$@"; do
    cd "$(dirname "$f")"
    '"${APLZ_PATH}"' compress "$f" 2>&1 | logger -t APLZ
    osascript -e "display notification \"圧縮完了: $(basename "$f")\" with title \"APLZ\""
done'

echo "==> Creating: APLZ で圧縮.workflow"
create_workflow "APLZ で圧縮" "$COMPRESS_SCRIPT"

# 解凍アクション: 選択された .aplz ファイルを解凍
EXTRACT_SCRIPT='for f in "$@"; do
    dir="$(dirname "$f")"
    '"${APLZ_PATH}"' extract "$f" "$dir" 2>&1 | logger -t APLZ
    osascript -e "display notification \"解凍完了: $(basename "$f")\" with title \"APLZ\""
done'

echo "==> Creating: APLZ で解凍.workflow"
create_workflow "APLZ で解凍" "$EXTRACT_SCRIPT"

# サービスキャッシュをリセット (変更を即座に反映)
/System/Library/CoreServices/pbs -flush 2>/dev/null || true

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
echo " 注意: 初回は「システム設定」→「プライバシーとセキュリティ」"
echo "       →「拡張機能」→「Finder」で有効化が必要な場合があります"
echo ""
echo " アンインストール:"
echo "   ./setup_finder.sh uninstall"
echo "============================================================"
