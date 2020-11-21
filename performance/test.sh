#!/bin/bash

export LEVEL=4
export NEWDIR="$(date '+%Y-%m-%d_%H-%M-%S').logs"

mkdir /root/performance/$NEWDIR

let NUMBER_OF_PUZZLES=1

for (( c=1; c<=LEVEL; c++ ))
do  

echo *****************************************
echo LEVEL = $c   PUZZLES = $NUMBER_OF_PUZZLES
echo *****************************************

#Generate sudokus with QQwing:
/usr/bin/time -v -a --output=/root/performance/$NEWDIR/qqwing_gen.log -p sh -c "java -jar /root/qqwing-1.3.4.jar --generate ${NUMBER_OF_PUZZLES} --difficulty expert --compact > /root/performance/${NEWDIR}/sudoku${NUMBER_OF_PUZZLES}.txt"

#Solve sudokus with QQwing (reference):
/usr/bin/time -v -a --output=/root/performance/$NEWDIR/sudoku-qqwing.log -p sh -c "cat /root/performance/$NEWDIR/sudoku${NUMBER_OF_PUZZLES}.txt | java -jar /root/qqwing-1.3.4.jar --solve --compact > /root/performance/${NEWDIR}/sudoku${NUMBER_OF_PUZZLES}_sol-qqwing.txt"

#Solve sudokus with rust sudoku:
/usr/bin/time -v -a --output=/root/performance/$NEWDIR/sudoku-rust.log -p sh -c "/root/sudoku-rust /root/performance/${NEWDIR}/sudoku${NUMBER_OF_PUZZLES}.txt /root/performance/${NEWDIR}/sudoku${NUMBER_OF_PUZZLES}_sol-rust.txt"

#Solve sudokus with scala sudoku:
/usr/bin/time -v -a --output=/root/performance/$NEWDIR/sudoku-scala.log -p sh -c "java -jar /root/sudoku-scala.jar /root/performance/${NEWDIR}/sudoku${NUMBER_OF_PUZZLES}.txt /root/performance/${NEWDIR}/sudoku${NUMBER_OF_PUZZLES}_sol-scala.txt"

#Solve sudokus with scalanative sudoku:
/usr/bin/time -v -a --output=/root/performance/$NEWDIR/sudoku-scalanative.log -p sh -c "/root/sudoku-scalanative /root/performance/${NEWDIR}/sudoku${NUMBER_OF_PUZZLES}.txt /root/performance/${NEWDIR}/sudoku${NUMBER_OF_PUZZLES}_sol-scalanative.txt"

let NUMBER_OF_PUZZLES=$(($NUMBER_OF_PUZZLES*10))

#Update result csvs:
scala /root/performance/prepare_data.sh /root/performance/${NEWDIR}

done

exit
