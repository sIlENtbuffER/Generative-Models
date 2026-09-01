import io
import os
import struct
import numpy as np
from PIL import Image

SRC_PATH = "data/CelebA/img_align_celeba.zip"
DST_PATH = "data/CelebA/celeba.bin"
CROP = 148
SIZE = 64
COUNT = 30000

def jpegs(path):
    end = os.path.getsize(path)
    with open(path, "rb") as f:
        while True:
            header = f.read(30)
            if len(header) < 30 or header[:4] != b"PK\x03\x04":
                return
            # Local file header: compressed size at 18, name and extra lengths at 26
            csize, nlen, elen = struct.unpack("<I", header[18:22]) + struct.unpack("<HH", header[26:30])
            name = f.read(nlen).decode()
            f.seek(elen, os.SEEK_CUR)
            if f.tell() + csize > end:
                return
            data = f.read(csize)
            if name.endswith(".jpg"):
                yield data

def main():
    written = 0

    with open(DST_PATH, "wb") as out:
        out.write(struct.pack("<IIII", COUNT, 3, SIZE, SIZE))

        for data in jpegs(SRC_PATH):
            if written == COUNT:
                break

            image = Image.open(io.BytesIO(data)).convert("RGB")
            left = (image.width - CROP) // 2
            top = (image.height - CROP) // 2
            image = image.crop((left, top, left + CROP, top + CROP)).resize((SIZE, SIZE), Image.BILINEAR)
            out.write(np.asarray(image, dtype=np.uint8).transpose(2, 0, 1).tobytes())

            written += 1
            if written % 10000 == 0:
                print(f"{written}/{COUNT}")

        out.seek(0)
        out.write(struct.pack("<IIII", written, 3, SIZE, SIZE))

    print(f"Wrote {written} images to {DST_PATH}")

if __name__ == "__main__":
    main()
