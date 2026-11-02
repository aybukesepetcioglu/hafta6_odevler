import cv2
import numpy as np

resim = cv2.imread("resim.jpg")
gri = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
kenar = cv2.Canny(gri, 100, 200)
kenar_median = cv2.medianBlur(kenar, 5)

yukseklik, genislik = kenar_median.shape

derece = np.arange(-90, 90, 1)
radyan = np.deg2rad(derece)

cos_tablo = np.cos(radyan)
sin_tablo = np.sin(radyan)

kosegen = int(np.ceil(np.sqrt(yukseklik ** 2 + genislik ** 2)))
rho_sayisi = 2 * kosegen + 1   

oy_matrisi = np.zeros((rho_sayisi, len(radyan)), dtype=np.int32)

ys, xs = np.nonzero(kenar_median)

for i in range(len(xs)):
    x = xs[i]
    y = ys[i]
    for t in range(len(radyan)):
        rho = x * cos_tablo[t] + y * sin_tablo[t]
        rho_index = int(round(rho)) + kosegen   
        oy_matrisi[rho_index, t] += 1            

print("Oylama matrisi boyutu:", oy_matrisi.shape)
print("En yuksek oy:", oy_matrisi.max())

oy_resim = (oy_matrisi / oy_matrisi.max() * 255).astype(np.uint8)
cv2.imwrite("5_oylama_matrisi.png", oy_resim)

oy_renkli = cv2.applyColorMap(oy_resim, cv2.COLORMAP_JET)
oy_renkli = cv2.resize(oy_renkli, None, fx=2, fy=1)
cv2.imwrite("5b_oylama_matrisi_renkli.png", oy_renkli)

ESIK = int(0.6 * oy_matrisi.max())
print("Esik deger:", ESIK)

rho_idx, t_idx = np.where(oy_matrisi >= ESIK)

secilen = []
for r, t in zip(rho_idx, t_idx):
    benzer = False
    for (r2, t2) in secilen:
        if abs(r - r2) < 10 and abs(t - t2) < 5:
            benzer = True
            break
    if not benzer:
        secilen.append((r, t))

sonuc = resim.copy()
for (r, t) in secilen:
    rho = r - kosegen
    theta = np.deg2rad(derece[t])
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(sonuc, (x1, y1), (x2, y2), (255, 0, 0), 2)   # MAVI

print("Bulunan dogru sayisi:", len(secilen))
cv2.imwrite("6_kendi_hough_sonuc.png", sonuc)