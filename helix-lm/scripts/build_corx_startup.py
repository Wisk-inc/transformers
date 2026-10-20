"""
Build the CorX5.5 deliverables (run: python scripts/build_corx_startup.py): the small startup script that
downloads the wheel from the model repository, the self-contained bundle with the wheel embedded (for pasting
over SSH), and the wheel itself, which is uploaded to the Hub repository under code/.
"""

import base64
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent
dist = REPO / "dist"
dist.mkdir(exist_ok=True)
for old in dist.glob("*.whl"):
    old.unlink()
subprocess.run([sys.executable, "-m", "pip", "wheel", str(REPO), "--no-deps", "-w", str(dist), "-q"], check=True)
wheel = next(dist.glob("helix_lm-*.whl"))
template = (HERE / "corx_startup.template.sh").read_text().replace("__WHEEL_NAME__", wheel.name)
(HERE / "corx_startup.sh").write_text(template)  # the small one: paste into the provider's startup field

encoded = base64.b64encode(wheel.read_bytes()).decode()
lines = "\n".join(encoded[i : i + 100] for i in range(0, len(encoded), 100))
# The bundle carries the wheel itself: the setup writes it to /root before the install script runs, so the
# install script's "already on disk" branch takes it and nothing is downloaded.
anchor = "cat > /root/corx_install.sh <<'CORX_INSTALL'\n"
assert template.count(anchor) == 1
embedded = template.replace(
    anchor,
    "base64 -d > /root/"
    + wheel.name
    + " <<'CORX_WHEEL'  # the package, embedded in this bundle\n"
    + lines
    + "\nCORX_WHEEL\n"
    + anchor,
)
(dist / "corx_startup_bundle.sh").write_text(embedded)  # the self-contained one: paste or scp over SSH
print(
    f"wheel {wheel.name}: {wheel.stat().st_size / 1024:.0f} KB; startup script {len(template) / 1024:.0f} KB; "
    f"bundle {len(embedded) / 1024:.0f} KB"
)
