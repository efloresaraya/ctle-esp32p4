"""Read V, A and W from the KWS-X1 display with the macOS Vision OCR.

The three main readings carry four decimals; the side panel (05.03V, D+ 3.14V,
...) carries two, so a four-decimal pattern isolates the main readings. Units
are used when the OCR keeps them attached, vertical order otherwise.
Every reading is checked by V x A = W.
"""
import re, Vision
from Foundation import NSData, NSURL

NUM = re.compile(r"(\d[.,]\d{4})\s*([VvAaWw])?")
# The large blue voltage digits sometimes lose their decimal point in the OCR
# ("49075V"). The bus is near 5 V, so five digits followed by V can only mean
# d.dddd; the V x A = W check still decides whether the frame is kept.
NUM_V_NODOT = re.compile(r"(?<![\d.,])(\d)(\d{4})\s*[Vv]\b")
TOL_W = 0.0002          # observed residuals are within 0.05 mW; one wrong digit in A moves V*A by 0.5 mW

def _observations(handler):
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(False)
    ok, err = handler.performRequests_error_([req], None)
    out = []
    for obs in (req.results() or []):
        text = obs.topCandidates_(1)[0].string()
        y = obs.boundingBox().origin.y          # normalized, origin bottom-left
        out.append((text, y))
    return out

def ocr_path(path):
    url = NSURL.fileURLWithPath_(path)
    return _observations(Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None))

def ocr_jpeg_bytes(b):
    data = NSData.dataWithBytes_length_(b, len(b))
    return _observations(Vision.VNImageRequestHandler.alloc().initWithData_options_(data, None))

def parse(observations):
    """Return dict(v, a, w, residual, ok) or None.

    Unit letters and on-screen position are not trusted: the OCR misreads 'W'
    as 'v' when the display is rotated relative to the camera. Instead the
    physics assigns the roles. Of the four-decimal numbers on screen, V is the
    one near the 5 V bus, and of the rest, W is the one that equals V x A.
    A misread digit breaks that identity, so the reading is rejected.
    """
    from itertools import permutations
    vals = [float(m.group(1).replace(",", ".")) for text, _ in observations for m in NUM.finditer(text)]
    vals += [float(f"{m.group(1)}.{m.group(2)}") for text, _ in observations
             for m in NUM_V_NODOT.finditer(text)]
    best = None
    for v, a, w in permutations(vals, 3):
        if not (4.0 <= v <= 5.6 and 0.0 < a < 0.5 and 0.2 <= w <= 1.5):
            continue
        res = v * a - w
        if best is None or abs(res) < abs(best[3]):
            best = (v, a, w, res)
    if best is None:
        return None
    v, a, w, res = best
    return dict(v=v, a=a, w=w, residual=res, ok=abs(res) <= TOL_W)
