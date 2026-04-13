# Párhuzamos eszközök programozása - OpenCL Mátrix Determináns

**Név:** Lengyel Gábor
**Neptun-kód:** GKIU70

## A projekt célja
A projekt célja nagyméretű négyzetes mátrixok determinánsának kiszámítása CPU-n és GPU-n, OpenCL keretrendszer felhasználásával. Két különböző megközelítés is implementálásra került, valamint összehasonlításra kerültek a futási idők.

## A repository felépítése

### feladat/gauss/
Ez a könyvtár a klasszikus **Gauss-elimináció** implementációját tartalmazza.

### feladat/lu_block/
Ez a könyvtár a **Blokkosított LU felbontás** implementációját tartalmazza.

### feladat/benchmarks.pdf
A különböző algoritmusok tesztelése, futási eredmények összehasonlítása.

Az algoritmusok leírása a saját jegyzékük README fájljában találhatók.

## Fordítás
A programok a make parancs kiadásával fordíthatók.
