import cv2
import numpy as np

resim = np.full((200, 200, 3), 255, dtype=np.uint8)
for r in (20, 35, 50):
    cv2.circle(resim, (100, 100), r, (0, 0, 0), 2)   
cv2.circle(resim, (40, 40), 25, (0, 0, 0), 2)        
cv2.imwrite("resim.jpg", resim)

gri = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
kenar = cv2.Canny(gri, 100, 200)

cv2.imwrite("t1_kenar.png", kenar)
yukseklik, genislik = kenar.shape

yaricaplar = np.arange(15, 61)          
derece = np.arange(0, 360, 5)           
radyan = np.deg2rad(derece)

cos_tablo = np.cos(radyan)
sin_tablo = np.sin(radyan)

ESIK = 40          
ES_MERKEZ_ESIK = 2 

oy = np.zeros((len(yaricaplar), yukseklik, genislik), dtype=np.int32)

halka_sayisi = np.zeros((yukseklik, genislik), dtype=np.int32)

ys, xs = np.nonzero(kenar)
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
                # 1) normal cember oyu
                oy[ri, b, a] += 1

                if oy[ri, b, a] == ESIK:
                    komsu = oy[max(ri - 5, 0):ri + 6, b, a]
                    if np.sum(komsu >= ESIK) == 1:      
                        halka_sayisi[b, a] += 1         

print("En yuksek cember oyu:", oy.max())
print("En cok halka sayisi:", halka_sayisi.max())

hs = (halka_sayisi / max(halka_sayisi.max(), 1) * 255).astype(np.uint8)
cv2.imwrite("t2_halka_sayisi_matrisi.png", cv2.resize(hs, None, fx=2, fy=2))

sonuc = resim.copy()
merkez_b, merkez_a = np.where(halka_sayisi >= ES_MERKEZ_ESIK)

cizilen_merkezler = []
for b, a in zip(merkez_b, merkez_a):
    yakin = False
    for (a2, b2) in cizilen_merkezler:
        if abs(a - a2) < 5 and abs(b - b2) < 5:
            yakin = True
            break
    if yakin:
        continue
    cizilen_merkezler.append((a, b))
    ri_listesi = np.where(oy[:, b, a] >= ESIK)[0]
    for ri in ri_listesi:
        cv2.circle(sonuc, (a, b), int(yaricaplar[ri]), (0, 0, 255), 1)   
    cv2.circle(sonuc, (a, b), 2, (0, 200, 0), -1)                        
    print("Es merkezli grup: merkez =", (int(a), int(b)), " yaricaplar =", yaricaplar[ri_listesi])

cv2.imwrite("t3_es_merkezli_sonuc.png", sonuc)