import logging
from config import Paths
Paths.LOG.mkdir(exist_ok=True)

logging.basicConfig(
    filename=Paths.LOG / "run.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)