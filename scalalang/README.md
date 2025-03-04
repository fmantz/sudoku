# README

* Build assembly `sbt clean assembly`
* Run by `java -jar ./target/scala-3.6.3/sudoku-assembly-1.0.2.jar ./src/test/resources/p096_sudoku.txt`
* Sudokus can be generated e.g. with <https://qqwing.com/download.html> `java -jar qqwing-1.3.4.jar --generate 1000 --difficulty expert --compact > sudoku.txt`
* Sudokus can be fast solved by QQWing `cat sudoku.txt | java -jar qqwing-1.3.4.jar --solve --compact > sudoku_sol.txt`
* Note in my tests QQWing unfortunately produces results that are not ordered accordingly the input puzzles but randomly (it's a multithreading application)

## Scala Native is again supported :-)
* Install requirements for Scala Native first: <https://scala-native.org/en/stable/user/setup.html#installing-clang-and-runtime-dependencies>
* Build native by `sbt -DNATIVE clean nativeLink`  (will last some time)
* Run native by `./target/scala-3.4.2/sudoku-out ./src/test/resources/p096_sudoku.txt`


## Build Scala Binary with Sbt + GraalVm 

```
sbt assembly
cp ./target/scala-3.4.0/sudoku-assembly-1.0.0.jar ./
docker run -it --rm --mount type=bind,source="$(pwd)",target=/app ghcr.io/graalvm/native-image-community:21 /app/sudoku -jar /app/sudoku-assembly-1.0.0.jar
```
