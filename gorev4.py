import cv2
import numpy as np

resim = cv2.imread("resim.jpg")

kucuk = cv2.resize(resim, None, fx=0.5, fy=0.5)
gri = cv2.cvtColor(kucuk, cv2.COLOR_BGR2GRAY)

kenar = cv2.Canny(gri, 100, 200)
kenar_median = cv2.medianBlur(kenar, 5)
cv2.imwrite("c4_kucuk_kenar.png", kenar_median)

yukseklik, genislik = kenar_median.shape

r_min = 8
r_max = 30
yaricaplar = np.arange(r_min, r_max + 1)

derece = np.arange(0, 360, 10)
radyan = np.deg2rad(derece)
cos_tablo = np.cos(radyan)
sin_tablo = np.sin(radyan)
oy = np.zeros((len(yaricaplar), yukseklik, genislik), dtype=np.int32)

ys, xs = np.nonzero(kenar_median)
print("Kenar piksel sayisi:", len(xs))

for i in range(len(xs)):
    x = xs[i]
    y = ys[i]
    for ri in range(len(yaricaplar)):
        r = yaricaplar[ri]
        for t in range(len(radyan)):
            a = int(round(x - r * cos_tablo[t]))
            b = int(round(y - r * sin_tablo[t]))
            if 0 <= a < genislik and 0 <= b < yukseklik:
                oy[ri, b, a] += 1

print("3 boyutlu oylama matrisi boyutu:", oy.shape)
print("En yuksek oy:", oy.max())

ESIK = int(0.7 * oy.max())
print("Esik deger:", ESIK)

ri_idx, b_idx, a_idx = np.where(oy >= ESIK)

secilen = []
for ri, b, a in zip(ri_idx, b_idx, a_idx):
    r = yaricaplar[ri]
    benzer = False
    for (a2, b2, r2) in secilen:
        if abs(a - a2) < 8 and abs(b - b2) < 8 and abs(r - r2) < 8:
            benzer = True
            break
    if not benzer:
        secilen.append((a, b, r))

sonuc = resim.copy()
for (a, b, r) in secilen:
    cv2.circle(sonuc, (a * 2, b * 2), r * 2, (255, 0, 0), 2)   # MAVI cember
    cv2.circle(sonuc, (a * 2, b * 2), 2, (0, 255, 255), 3)     # SARI merkez

print("Bulunan cember sayisi:", len(secilen))
cv2.imwrite("c5_kendi_hough_cember.png", sonuc)

en_iyi_ri = np.unravel_index(np.argmax(oy), oy.shape)[0]
kesit = oy[en_iyi_ri]
kesit_resim = (kesit / max(kesit.max(), 1) * 255).astype(np.uint8)
cv2.imwrite("c6_oylama_kesiti.png", cv2.resize(kesit_resim, None, fx=2, fy=2))