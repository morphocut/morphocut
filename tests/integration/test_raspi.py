import pytest
from morphocut.integration.raspi import PiCameraReader, is_raspberrypi
from morphocut import Pipeline
from morphocut._optional import UnavailableObjectError
from morphocut.stream import Slice

picamera_available = False

try:
    import picamera
    picamera.PiCamera()
except:
    pass
else:
    picamera_available = True

print("picamera_available", picamera_available)
    

@pytest.mark.skipif(picamera_available, reason="picamera available")
def test_picamera_available():
    with Pipeline() as p:
        frame = PiCameraReader()

    with pytest.raises(UnavailableObjectError):
        p.run()

@pytest.mark.skipif(not picamera_available, reason="picamera not available")
def test_PiCamera():
    with Pipeline() as p:
        frame = PiCameraReader()

        # Only capture 10 frames
        Slice(10)

    p.run()