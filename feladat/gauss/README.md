# Gauss-elimináció

Ez a könyvtár a projekt Gauss-elimináción alapuló megoldásait tartalmazza. A program a mátrix determinánsát számolja ki szekvenciális (CPU) és párhuzamosított (GPU) megközelítésben.

## Az algoritmusok működése

A Gauss-elimináció egy széles körben alkalmazott lineáris algebrai módszer. A determináns kiszámításának kontextusában az algoritmus célja, hogy az eredeti négyzetes mátrixot elemi sorműveletek (egy sor konstansszorosának kivonása egy másikból, illetve sorok cseréje) segítségével felső háromszögmátrixszá alakítsa. Ez azért hasznos, mert egy háromszögmátrix determinánsa megegyezik a főátlóban lévő elemek szorzatával.

### 1. CPU Implementáció (Szekvenciális)
A `matrix.c` fájlban található algoritmus a klasszikus Gauss-eliminációt valósítja meg részleges főelem-kiválasztással.
* **Főelem-kiválasztás:** Minden eliminációs lépés előtt az algoritmus megkeresi az aktuális oszlopban a legnagyobb abszolút értékű elemet a numerikus stabilitás érdekében. Ha sorcserére van szükség, a determináns előjelét (`sign`) megváltoztatja.
* **Determináns számítás:** A felső háromszögmátrixszá alakítás után a főátló elemeit szorozza össze. A lebegőpontos túlcsordulás elkerülésére a szorzatot normalizált mantissza és kitevő formájában kezeli a rendszer.

### 2. GPU Implementáció (OpenCL)
A párhuzamosított verzió közvetlenül a videókártya globális memóriáját (VRAM) használja. A számítás fázisait a CPU vezérli egy cikluson keresztül, amely lépésenként két OpenCL kernelt indít:

* **`pivot_and_swap` kernel:** Mivel a maximumkeresés szekvenciális feladat, ez a kernel egyetlen szálon fut le a GPU-n. Megkeresi az oszlop maximumát, elvégzi a memóriában a sorcserét, és frissíti a determináns előjelét a globális memóriában.
* **`calculate_determinant_gauss` kernel:** Ez végzi a nehéz számítási munkát egy kétdimenziós munkaterületen. Minden GPU szál egyetlen elem frissítéséért felelős. 
* **Végeredmény kiszámítása (CPU oldalon):** Az elimináció befejezése után a felső háromszögmátrixszá alakított adatok visszakerülnek a processzorhoz (RAM). A determináns tényleges kiszámítását (a főátló elemeinek összeszorzását és a mantissza/kitevő normalizálását) a CPU végzi el a visszakapott adatokból, figyelembe véve a GPU által számontartott előjelváltozásokat.

## A könyvtár fájljai

* `main.c`: A benchmark futtatásáért, a CPU és GPU idők, valamint a relatív hiba kiszámításáért felelős.
* `matrix.c` / `matrix.h`: A CPU-s számítási logika, illetve az OpenCL keretrendszer inicializálása, a memóriafoglalás és a kernelek paraméterezése.
* `kernel/sample.cl`: A videókártyán futó OpenCL kernel kódok.
* `test_determinant.c`: CMocka alapú egységtesztek a numerikus pontosság ellenőrzésére.
* `file.c` / `file.h` és `kernel_loader.c`: Segédfüggvények a mérési eredmények lementéséhez és az OpenCL forráskód betöltéséhez.

## Fordítás és futtatás

A projekt fordítását a gyökérkönyvtárban kiadott `make` paranccsal végezhetjük el. Ez automatikusan legenerálja a főprogramot (`main.exe`) és az egységteszteket (`test_determinant.exe`) is.

A fő benchmark program indítása, paraméterként megadható mátrix mérettel:
```bash
.\main.exe 4000