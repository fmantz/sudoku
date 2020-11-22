# README

* Build assembly `cargo build --release`
* Run by `./target/release/sudoku ./test/resources/p096_sudoku.txt`
* Sudokus can be generated e.g. with https://qqwing.com/download.html `java -jar qqwing-1.3.4.jar --generate 1000 --difficulty expert --compact > sudoku.txt`
* Sudokus can be fast solved by QQWing `cat sudoku.txt | java -jar qqwing-1.3.4.jar --solve --compact > sudoku_sol.txt`
* Note in my tests QQWing unfortunately produces results that are not ordered accordingly the input puzzles but randomly (it's a multithreading application)
