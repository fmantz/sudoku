#!/bin/bash

export NEWDIR="$(date +"%Y_%m_%d_%I_%M_%p").logs"

mkdir ./performance/$NEWDIR

#Generate sudokus with QQwing:
/usr/bin/time -v -a --output=./performance/$NEWDIR/qqwing_gen.log -p sh -c "java -jar /root/qqwing-1.3.4.jar --generate 10 --difficulty expert --compact > ./performance/${NEWDIR}/sudoku10.txt"

#Solve sudokus with QQwing (reference):
/usr/bin/time -v -a --output=./performance/$NEWDIR/sudoku-qqwing.log -p sh -c "cat ./performance/$NEWDIR/sudoku10.txt | java -jar /root/qqwing-1.3.4.jar --solve --compact ./performance/${NEWDIR}/sudoku10.txt > ./performance/${NEWDIR}/sudoku10_sol-qqwing.txt"

#Solve sudokus with scala sudoku:
/usr/bin/time -v -a --output=./performance/$NEWDIR/sudoku-scala.log -p sh -c "java -jar /root/sudoku-scala.jar ./performance/${NEWDIR}/sudoku10.txt ./performance/${NEWDIR}/sudoku10_sol-scala.txt"

#Solve sudokus with scalanative sudoku:
/usr/bin/time -v -a --output=./performance/$NEWDIR/sudoku-scalanative.log -p sh -c "/root/sudoku-scalanative ./performance/${NEWDIR}/sudoku10.txt ./performance/${NEWDIR}/sudoku10_sol-scalanative.txt"

#Solve sudokus with rust sudoku:
/usr/bin/time -v -a --output=./performance/$NEWDIR/sudoku-rust.log -p sh -c "/root/sudoku-rust ./performance/${NEWDIR}/sudoku10.txt ./performance/${NEWDIR}/sudoku10_sol-rust.txt"

exit
