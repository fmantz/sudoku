# README

* Build assembly `cargo build --release`
* Run by `./target/release/sudoku ./test/resources/p096_sudoku.txt`
* Sudokus can be generated e.g. with <https://qqwing.com/download.html> `java -jar qqwing-1.3.4.jar --generate 1000 --difficulty expert --compact > sudoku.txt`
* Sudokus can be fast solved by QQWing `cat sudoku.txt | java -jar qqwing-1.3.4.jar --solve --compact > sudoku_sol.txt`
* Note in my tests QQWing unfortunately produces results that are not ordered accordingly the input puzzles but randomly (it's a multithreading application)

//TODO:
Notes:

- use CUDA 8 since my Grafic card does not support higher versions (compute capability 2.1)
- https://en.wikipedia.org/wiki/CUDA
- use LTS Kernel 5.4 since Kernels > 5.8 are unsupported by cuda
- install opencl-nvidia-390xx with headers (first!) 
- install cuda-8.0 mit yay

## Make
- make    //to build cuda so
- cargo build //to build rust file //cuda lib must be in a folder called /lib  next to /release oder /debug
- target\lib
- target\release
- target\debug
