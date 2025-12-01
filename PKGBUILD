# Maintainer: Will Handley <wh260@cantab.ac.uk>
pkgname=linux-voice
pkgver=0.1.0
pkgrel=1
pkgdesc="Voice-to-text dictation tool for Linux (X11) using OpenAI Whisper"
arch=('any')
url="https://github.com/williamjameshandley/linux-voice"
license=('MIT')
depends=(
    'python>=3.11'
    'python-numpy'
    'python-openai'
    'python-pynput'
    'python-sounddevice'
    'xdotool'
)
optdepends=(
    'sof-firmware: for internal microphone support on modern AMD laptops'
)
source=(
    'linux-voice.py'
    'linux-voice.service'
    'README.md'
    'LICENSE'
)
sha256sums=('SKIP' 'SKIP' 'SKIP' 'SKIP')

package() {

    # Install main script
    install -Dm755 linux-voice.py "$pkgdir/usr/bin/linux-voice"

    # Install systemd user service
    install -Dm644 linux-voice.service "$pkgdir/usr/lib/systemd/user/linux-voice.service"

    # Install documentation
    install -Dm644 README.md "$pkgdir/usr/share/doc/$pkgname/README.md"

    # Install license
    install -Dm644 LICENSE "$pkgdir/usr/share/licenses/$pkgname/LICENSE"
}
