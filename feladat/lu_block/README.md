# Blokkosított LU-felbontás

Ez a könyvtár a projekt cache-optimalizált, blokkosított (tiled) LU-felbontáson alapuló megoldását tartalmazza. Ez a megközelítés a naiv Gauss-elimináció sávszélesség-korlátos (Memory Bound) problémáját küszöböli ki azáltal, hogy drasztikusan csökkenti a globális videómemória (VRAM) írási és olvasási műveleteit.

## Az algoritmusok működése

A blokkosított LU-felbontás lényege, hogy a hatalmas mátrixot nem soronként, hanem fix méretű (például 128x128-as) részmátrixokra, úgynevezett csempékre (blokkokra) bontja. Ennek hatalmas előnye a GPU architektúrán, hogy egy-egy ilyen blokk befér a videókártya szupergyors lokális memóriájába (`__local`), így a számítások nagy része a lassú VRAM érintése nélkül végezhető el.

A determináns kiszámításának matematikai elve itt is az, hogy a mátrixot felső háromszögmátrixszá alakítjuk, majd a főátló elemeit összeszorozzuk.

### 1. CPU Implementáció (Referencia)
A `matrix.c` fájlban található CPU algoritmus referenciaként továbbra is a klasszikus Gauss-eliminációt használja részleges főelem-kiválasztással. Ez szolgál alapul a GPU-s LU-felbontás pontosságának (relatív hiba) és sebességének validálásához.

### 2. GPU Implementáció (OpenCL)
Az OpenCL alapú LU-felbontás három különálló, egymásra épülő kernel futtatásával dolgozza fel a mátrixot blokkról blokkra lépkedve. A CPU egy külső ciklusból vezérli a fázisokat:

* **`lu_factorize_block` kernel:** Ez a kernel az aktuális főátlón lévő blokkot (csempét) tölti be a GPU szupergyors lokális memóriájába (`__local`). Ezen a kis memóriaterületen végzi el a faktorizációt, majd az eredményt visszamentia globális memóriába.
* **`lu_update_panels` kernel:** Miután a diagonális blokk faktorizálása megtörtént, ez a kernel frissíti az aktuális blokk alatti (alsó panel) és melletti (jobb panel) részmátrixokat.
* **`lu_update_trailing_matrix` kernel:** Ez végzi a mátrix hátralévő részének (a frissített panelek alatti és jobbra eső területek) módosítását. Matematikailag ez a leginkább számításigényes fázis ($O(N^3)$ művelet).
* **Végeredmény kiszámítása (CPU oldalon):** A feldolgozás végén a felső háromszögmátrix alakot öltött adatok visszakerülnek a rendszer memóriájába. A determináns végső értékét a CPU számolja ki a főátló elemeinek összeszorzásával és szabványos mantissza/kitevő formátumra hozásával.

## A könyvtár fájljai

* `main.c`: A benchmark futtatásáért, a processzoros referenciamérésért, illetve az OpenCL eredmény validálásáért felel.
* `matrix.c` / `matrix.h`: A CPU-s számítási logika, a GPU kernelek futásidejű paraméterezése és a blokk-ciklusok vezérlése.
* `kernel/sample.cl`: A videókártyán futó három OpenCL kernel implementációja.
* `test_determinant.c`: CMocka alapú egységtesztek a numerikus pontosság ellenőrzésére.
* `file.c` / `file.h` és `kernel_loader.c`: Segédfüggvények a futási idők kiíratásához és a kernel forráskód beolvasásához.

## Fordítás és futtatás

A projekt fordítását a gyökérkönyvtárban kiadott `make` paranccsal végezhetjük el. Ez automatikusan legenerálja a főprogramot (`main.exe`) és az egységteszteket (`test_determinant.exe`) is.

A fő benchmark program indítása, paraméterként megadható mátrix mérettel:
```bash
.\main.exe 4000