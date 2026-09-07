"""ezcal - Easy Calculation.

コマンド一つで第一原理計算のワークフローを実行するツール。既定のエンジンは
Quantum ESPRESSO。エンジン層はプラグイン構造になっており、CLI やワークフローに
手を入れずに VASP や MLIP (SevenNet) へ差し替えられる。
"""

__version__ = "0.2.0"

from ezcal.config import Config, load_config  # noqa: E402,F401

__all__ = ["Config", "load_config", "__version__"]
