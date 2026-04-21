# Párhuzamos eszközök programozása - OpenCL mátrix determináns

**Név:** Lengyel Gábor  
**Neptun-kód:** GKIU70  

## A projekt célja
A projekt célja nagyméretű négyzetes mátrixok determinánsának hatékony kiszámítása CPU-n és GPU-n, OpenCL keretrendszer felhasználásával. A feladat során két különböző megközelítés is implementálásra került.

## A repository felépítése

* **`feladat/gauss/`**
Ez a könyvtár a Gauss-elimináció CPU-s és OpenCL implementációját tartalmazza.

* **`feladat/lu_block/`**
Ez a könyvtár a cache-optimalizált, blokkosított LU-felbontás implementációját tartalmazza, amely a lokális memória kihasználásával jelentős sebességnövekedést ér el.

* **`feladat/benchmark.pdf`**
A különböző algoritmusok tesztelésének, teljesítményük mérésének és a futási eredmények összehasonlításának dokumentációja.

*(Az algoritmusok dokumentációi az adott almappák README fájljaiban találhatók.)*