#!/usr/bin/env bash
#
# Prognosis 一键安装脚本
#
# 用法:
#   bash <(curl -fsSL https://raw.githubusercontent.com/Mirakelor/Prognosis/master/install.sh)
#   或
#   curl -fsSL -o install.sh https://raw.githubusercontent.com/Mirakelor/Prognosis/master/install.sh && bash install.sh
#
# 行为:
#   1. 优先下载 GitHub Releases 预编译二进制（零依赖、无需 sudo、无需 Rust）
#   2. 无匹配发布时自动从源码构建:
#      - rustup 缺失时自动安装（用户级，无需 sudo）
#      - C linker (cc/gcc/clang) 缺失时自动安装（Linux 需一次 sudo apt）
#      - clone 源码 -> cargo build --release -> 安装
#   3. 安装到 ~/.local/bin（可用 PROGNOSIS_INSTALL_DIR 覆盖）
#   4. PATH 不含安装目录时自动追加到当前 shell 的 rc 文件
#
# 环境变量:
#   PROGNOSIS_INSTALL_DIR   安装目录（默认 ~/.local/bin）
#   PROGNOSIS_REPO          仓库（默认 Mirakelor/Prognosis）
#   PROGNOSIS_SKIP_RELEASE  设为 1 跳过二进制下载直接源码构建
#   PROGNOSIS_SKIP_BUILD_DEPS 设为 1 不自动安装编译器（构建缺 cc 时直接报错）

set -euo pipefail

REPO="${PROGNOSIS_REPO:-Mirakelor/Prognosis}"
INSTALL_DIR="${PROGNOSIS_INSTALL_DIR:-$HOME/.local/bin}"
BIN_NAME="prognosis"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

say()  { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- 1. 平台检测 ----------
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
  Linux)  OS="linux" ;;
  Darwin) OS="macos" ;;
  *) die "unsupported OS: $OS (only Linux and macOS are supported)" ;;
esac
case "$ARCH" in
  x86_64|amd64)  ARCH="x86_64" ;;
  aarch64|arm64) ARCH="aarch64" ;;
  *) die "unsupported architecture: $ARCH (only x86_64 and aarch64 are supported)" ;;
esac
say "Detected $OS/$ARCH"

command -v curl >/dev/null 2>&1 || die "curl is required but not found"
command -v mkdir install >/dev/null 2>&1 || die "coreutils are required but not found"

ensure_path() {
  case ":$PATH:" in
    *":$INSTALL_DIR:"*) return 0 ;;
  esac
  local rc=""
  case "${SHELL:-}" in
    *zsh)  rc="$HOME/.zshrc" ;;
    *bash) rc="$HOME/.bashrc" ;;
    *)     rc="$HOME/.profile" ;;
  esac
  if [ -n "$rc" ] && [ -w "$rc" ]; then
    printf '\nexport PATH="%s:$PATH"\n' "$INSTALL_DIR" >> "$rc"
    warn "Added '$INSTALL_DIR' to PATH in $rc (restart your shell or run: export PATH=\"$INSTALL_DIR:\$PATH\")"
  else
    warn "Install directory '$INSTALL_DIR' is not on your PATH."
    warn "Add it manually: export PATH=\"$INSTALL_DIR:\$PATH\""
  fi
}

print_done() {
  say "Prognosis installed successfully!"
  printf '  Run \033[1mprognosis\033[0m to start, then add a model with \033[1m/models\033[0m\n'
  printf '  Docs: https://github.com/%s#readme\n' "$REPO"
}

# ---------- 2. 尝试 GitHub Releases 预编译二进制 ----------
install_release() {
  local url="$1"
  say "Downloading release binary: $url"
  curl -fsSL "$url" -o "$TMP_DIR/prognosis.tar.gz" \
    || die "failed to download release binary"
  tar -xzf "$TMP_DIR/prognosis.tar.gz" -C "$TMP_DIR" \
    || die "failed to extract release binary"
  [ -f "$TMP_DIR/$BIN_NAME" ] || die "release archive does not contain $BIN_NAME"
  mkdir -p "$INSTALL_DIR"
  install -m755 "$TMP_DIR/$BIN_NAME" "$INSTALL_DIR/$BIN_NAME"
  return 0
}

if [ "${PROGNOSIS_SKIP_RELEASE:-0}" != "1" ]; then
  ASSET="prognosis-$OS-$ARCH.tar.gz"
  say "Looking for a prebuilt binary ($ASSET) in GitHub Releases..."
  API="https://api.github.com/repos/$REPO/releases/latest"
  RELEASE_JSON="$(curl -fsSL --max-time 15 "$API" 2>/dev/null || true)"
  if [ -n "$RELEASE_JSON" ]; then
    DOWNLOAD_URL="$(printf '%s' "$RELEASE_JSON" \
      | grep -o '"browser_download_url": *"[^"]*"' \
      | sed "s/.*\"\(.*\)\"/\1/" \
      | grep "/$ASSET$" | head -n1 || true)"
    if [ -n "$DOWNLOAD_URL" ]; then
      install_release "$DOWNLOAD_URL"
      say "Installed $BIN_NAME to $INSTALL_DIR (prebuilt binary)"
      ensure_path
      print_done
      exit 0
    fi
    warn "No prebuilt binary for $OS/$ARCH; falling back to source build"
  else
    warn "Could not reach GitHub API; falling back to source build"
  fi
fi

# ---------- 3. 源码构建兜底 ----------
say "Source build path"

# 3.1 rustup / cargo
if ! command -v cargo >/dev/null 2>&1; then
  if command -v rustup >/dev/null 2>&1; then
    say "cargo missing but rustup present; installing toolchain (minimal profile)"
    rustup toolchain install stable --profile minimal
  else
    say "Installing rustup (user-level, no sudo needed)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal \
      || die "rustup installation failed; install it manually from https://rustup.rs"
  fi
  # shellcheck disable=SC1091
  [ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
fi
command -v cargo >/dev/null 2>&1 || die "cargo still not available after rustup install"

# 3.2 C linker (cc/gcc/clang)
if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then
  if [ "${PROGNOSIS_SKIP_BUILD_DEPS:-0}" = "1" ]; then
    die "no C compiler found (cc/gcc/clang); install build-essential (Debian/Ubuntu) or Xcode Command Line Tools (macOS) and re-run"
  fi
  if [ "$OS" = "linux" ]; then
    if command -v apt-get >/dev/null 2>&1; then
      say "No C compiler found; installing build-essential (requires sudo once)"
      sudo apt-get update -qq && sudo apt-get install -y --no-install-recommends build-essential \
        || die "failed to install build-essential; run: sudo apt-get install -y build-essential"
    else
      die "no C compiler found; install a C toolchain for your distribution (e.g. 'sudo dnf install gcc' or 'sudo pacman -S base-devel') and re-run"
    fi
  else
    warn "No C compiler found; macOS needs Xcode Command Line Tools:"
    warn "  xcode-select --install"
    warn "After installing, re-run this script."
    exit 1
  fi
fi

# 3.3 clone + build + install
if ! command -v git >/dev/null 2>&1; then
  if [ "$OS" = "linux" ] && command -v apt-get >/dev/null 2>&1; then
    say "Installing git (requires sudo once)"
    sudo apt-get update -qq && sudo apt-get install -y --no-install-recommends git \
      || die "failed to install git"
  else
    die "git is required but not found"
  fi
fi
say "Cloning $REPO"
git clone --depth 1 "https://github.com/$REPO.git" "$TMP_DIR/src" \
  || die "git clone failed; check the repository URL and your network"
say "Building release binary (this can take a few minutes)"
( cd "$TMP_DIR/src" && cargo build --release ) || die "cargo build failed"
[ -f "$TMP_DIR/src/target/release/$BIN_NAME" ] || die "build did not produce $BIN_NAME"
mkdir -p "$INSTALL_DIR"
install -m755 "$TMP_DIR/src/target/release/$BIN_NAME" "$INSTALL_DIR/$BIN_NAME"
say "Installed $BIN_NAME to $INSTALL_DIR (built from source)"

ensure_path
print_done
