all: main test

main:
	gcc main.c src/matrix.c src/file.c src/kernel_loader.c -o main.exe -Iinclude -lOpenCL -lm

test:
	gcc tests/test_determinant.c src/matrix.c src/file.c src/kernel_loader.c -o test_determinant.exe -Iinclude -lOpenCL -lcmocka -lm