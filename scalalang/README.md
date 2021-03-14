# README

* Build assembly `sbt clean assembly`
* Run by `java -jar ./target/scala-2.11/sudoku-assembly-0.7.0.jar ./src/test/resources/p096_sudoku.txt`
* Sudokus can be generated e.g. with <https://qqwing.com/download.html> `java -jar qqwing-1.3.4.jar --generate 1000 --difficulty expert --compact > sudoku.txt`
* Sudokus can be fast solved by QQWing `cat sudoku.txt | java -jar qqwing-1.3.4.jar --solve --compact > sudoku_sol.txt`
* Note in my tests QQWing unfortunately produces results that are not ordered accordingly the input puzzles but randomly (it's a multithreading application)

## Scala Native is only supported to Version 0.2
* Install requirements for Scala Native first: <https://scala-native.readthedocs.io/en/v0.3.9-docs/user/sbt.html>
* Build native by `sbt -DNATIVE clean nativeLink`  (will last some time)
* Run native by `./target/scala-2.11/sudoku-out ./src/test/resources/p096_sudoku.txt`
