BUILD
=====

* Install requirements for Scala Native first: https://scala-native.readthedocs.io/en/v0.3.9-docs/user/sbt.html
* Compile by "sbt clean nativeLink"  (will last some time)
* Run native file by "./target/scala-2.11/sudoku-out ./src/main/resources/p096_sudoku.txt"
* Sudokus can be generated e.g. with https://qqwing.com/download.html "java -jar qqwing-1.3.4.jar --generate 1000 --difficulty expert --compact > sudoku.txt"
* Sodokus can be fast solved by qqwing "time sudoku.txt | java -jar qqwing-1.3.4.jar --solve --compact > soduko_sol.txt"
