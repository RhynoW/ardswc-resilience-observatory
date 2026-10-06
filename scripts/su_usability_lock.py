"""把實務人員測試組與專家盲判表加密後放進 repo（webapp/change_detect_viewer/private_kits/），由 Flask 路由 /test 在輸入 4 碼存取碼後才解密提供。
存取碼只在執行時由環境變數 TEST_CODE 提供，不寫入任何檔案、不進 git。
加密：PBKDF2-HMAC-SHA256（ITER 次）→ 64 位元組金鑰（前 32 加密、後 32 驗證）；每檔 16 位元組隨機 nonce；以 HMAC-SHA256(金鑰, nonce‖計數器) 產生金鑰流與明文 XOR；
      HMAC-SHA256 對 nonce‖密文做驗證（encrypt-then-MAC）；輸出 base64 文字檔（HF 不收未走 LFS 的二進位檔）。只用標準函式庫，伺服器端不需額外套件。
安全性說明（重要）：repo 是公開的，4 碼數字只有 10,000 種可能——能取得 repo 的人可離線猜碼（約 ITER 次 PBKDF2 × 10,000 次，單核心數十分鐘）。
      這是「擋一般訪客、避免洩題」等級的門檻，不是機密保護；測試內容本身只有影像、公開資料與題目，不含個資。要更強請改用較長的存取碼或私有儲存。
用法：TEST_CODE=1234 python scripts/su_usability_lock.py
"""
import base64
import hashlib
import hmac
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "biggis_interp" / "poc" / "su_usability"
DST = REPO / "webapp" / "change_detect_viewer" / "private_kits"
ITER = 300_000
FILES = {"G1": "kit_G1.html", "G2": "kit_G2.html", "expert": "expert_review_20.html"}


def derive(code, salt, iters=ITER):
    dk = hashlib.pbkdf2_hmac("sha256", code.encode(), salt, iters, 64)
    return dk[:32], dk[32:]


def _ks(ek, nonce, n):
    out = bytearray()
    c = 0
    while len(out) < n:
        out += hmac.new(ek, nonce + c.to_bytes(8, "big"), hashlib.sha256).digest()
        c += 1
    return bytes(out[:n])


def encrypt(plain, ek, mk):
    nonce = os.urandom(16)
    ks = _ks(ek, nonce, len(plain))
    ct = bytes(a ^ b for a, b in zip(plain, ks))
    mac = hmac.new(mk, nonce + ct, hashlib.sha256).digest()
    return base64.b64encode(nonce + ct + mac)


def decrypt(blob_b64, ek, mk):
    raw = base64.b64decode(blob_b64)
    nonce, ct, mac = raw[:16], raw[16:-32], raw[-32:]
    if not hmac.compare_digest(hmac.new(mk, nonce + ct, hashlib.sha256).digest(), mac):
        raise ValueError("bad mac")
    ks = _ks(ek, nonce, len(ct))
    return bytes(a ^ b for a, b in zip(ct, ks))


def main():
    code = os.environ.get("TEST_CODE", "")
    if not (len(code) == 4 and code.isdigit()):
        sys.exit("請以環境變數 TEST_CODE 提供 4 碼數字")
    DST.mkdir(exist_ok=True)
    salt = os.urandom(16)
    ek, mk = derive(code, salt)
    man = {"v": 1, "kdf": "pbkdf2-hmac-sha256", "iter": ITER, "salt": base64.b64encode(salt).decode(), "files": {}}
    (DST / "check.b64").write_bytes(encrypt(b"OK", ek, mk))
    man["files"]["check"] = "check.b64"
    for name, f in FILES.items():
        (DST / f"{name}.b64").write_bytes(encrypt((SRC / f).read_bytes(), ek, mk))
        man["files"][name] = f"{name}.b64"
    (DST / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8", newline="\n")
    for p in sorted(DST.iterdir()):
        print(p.name, round(p.stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
