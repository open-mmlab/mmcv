import base64
import os

for img in os.listdir('./'):
    if img.endswith('.png') or img.endswith('.jpg'):
        f = open(img, 'rb')
        ls_f=base64.b64encode(f.read())
        f.close()
        print(img)
        print(ls_f)