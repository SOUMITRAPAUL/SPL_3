from dataclasses import dataclass
from PIL import Image

@dataclass
class LowLightImage:
    """Represents a low-light input image."""
    data: Image.Image
    format: str

    def getMetadata(self):
        return {
            "size": self.data.size,
            "mode": self.data.mode,
            "format": self.format
        }

    def validateImage(self):
        # Basic validation
        return self.data is not None and self.data.mode == "RGB"

@dataclass
class EnhancedImage:
    """Represents the output of the enhancement process."""
    filePath: str = None
    data: Image.Image = None
    qualityMetrics: dict = None

    def displaySideBySide(self, original: LowLightImage):
        # Placeholder for UI display logic
        pass

    def exportImage(self, path: str):
        if self.data:
            self.data.save(path)
            self.filePath = path

    def computeMetrics(self):
        # Placeholder for PSNR/SSIM calculation
        self.qualityMetrics = {"status": "computed"}
