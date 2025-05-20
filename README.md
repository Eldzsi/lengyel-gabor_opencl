# Párhuzamos eszközök programozása féléves és gyakorlati feladatok


**Név:** Lengyel Gábor

**Neptun-kód:** GKIU70

## A repository szerkezete

A **gyakorlati feladatok** megoldásai a **gyakorlatok/** jegyzékben találhatók. Minden gyakorlat, és azon belül minden feladat külön jegyzékbe került.

A **féléves feladat** megoldása a **feladat/** jegyzékben található.

A make parancs kiadásával egyszerűen fordíthatók a programok.

## Féléves feladat specifikáció

A feladat célja egy n*n-es mátrix determinánsának kiszámítása többféle módon, valamint ezek teljesítményének összehasonlítása. A programnak a következő funkciókat kell megvalósítania:

- **Mátrix generálás:** Véletlenszerű egész értékekből álló négyzetes mátrix előállítása tetszőleges méretben.
- **Determináns számítás:**  
    - Rekurzív Laplace-kifejtés (CPU)
    - Iteratív Laplace-kifejtés (CPU)
    - OpenCL-alapú (GPU) számítás, ahol az iteratív Laplace-kifejtés kerül párhuzamosításra 
- **Időmérés és összehasonlítás:** A különböző módszerek futási idejének mérése és kiírása fájlba.

